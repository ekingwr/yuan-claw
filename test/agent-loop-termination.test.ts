/**
 * Agent 循环的终止路径与事件契约。
 *
 * 前端 useChat 完全依赖这些事件来复位 isProcessing、清理占位消息：
 *   - 正常收尾：assistant → run_end(final)
 *   - 超出步数：run_end(max_steps_exceeded)，**没有** assistant
 *   - 模型/确认阶段致命错误：run_error，**没有** run_end
 *   - 解析失败：run_error(parse_agent_response) 后循环继续，不是终止信号
 *
 * 这里把四条路径钉住，避免后端改动悄悄破坏前端的收尾逻辑。
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { runLocalAgentLoop } from "../src/agent/run-local-agent-loop.js";
import type { AgentEvent, EventBus } from "../src/events/event-bus.js";
import type { Tool, ToolResult } from "../src/tools/types.js";
import type { ChatMessage } from "../src/memory/types.js";

function makeBus(events: AgentEvent[]): EventBus {
    return {
        emit: (e) => events.push(e),
        showProcessing() {},
        hideProcessing() {},
    };
}

function textOf(message: ChatMessage | undefined): string {
    if (!message) return "";
    return typeof message.content === "string"
        ? message.content
        : message.content.map((p) => (p.type === "text" ? p.text : "[图片]")).join("");
}

const ECHO_TOOL: Tool = {
    name: "echo",
    description: "fake echo",
    riskLevel: "safe",
    async execute(): Promise<ToolResult> {
        return { success: true, output: "ok" };
    },
};

const PLANNER_RESPONSE = JSON.stringify({
    taskAnalysis: "t",
    selectedSkills: [],
    needsTools: true,
    plan: "p",
});

/**
 * 规划调用与协调者调用共用一个 modelClient，靠 system prompt 区分。
 * 规划固定返回「不选任何技能」，其余交给 coordinator 回调，参数是协调者的第几次调用。
 */
function wrapModel(
    coordinator: (callIndex: number) => string,
): { generate: (messages: ChatMessage[]) => Promise<string> } {
    let n = 0;
    return {
        async generate(messages: ChatMessage[]): Promise<string> {
            if (textOf(messages[0]).includes("任务规划助手")) return PLANNER_RESPONSE;
            n += 1;
            return coordinator(n);
        },
    };
}

function run(opts: {
    model: { generate: (messages: ChatMessage[]) => Promise<string> };
    maxSteps: number;
    events: AgentEvent[];
}) {
    return runLocalAgentLoop({
        userInput: "测试任务",
        modelClient: opts.model,
        tools: new Map([["echo", ECHO_TOOL]]),
        eventBus: makeBus(opts.events),
        maxSteps: opts.maxSteps,
        previousMessages: [],
        depth: 0,
    });
}

type RunEndEvent = Extract<AgentEvent, { type: "run_end" }>;
type RunErrorEvent = Extract<AgentEvent, { type: "run_error" }>;

const runEnds = (events: AgentEvent[]): RunEndEvent[] =>
    events.filter((e): e is RunEndEvent => e.type === "run_end");

const runErrors = (events: AgentEvent[]): RunErrorEvent[] =>
    events.filter((e): e is RunErrorEvent => e.type === "run_error");

const countType = (events: AgentEvent[], type: AgentEvent["type"]): number =>
    events.filter((e) => e.type === type).length;

const TOOL_CALL_ECHO = JSON.stringify({
    type: "tool_call",
    toolName: "echo",
    args: {},
});

test("正常收尾：run_start 只发一次，assistant 先于 run_end(final)", async () => {
    const events: AgentEvent[] = [];
    const model = wrapModel((n) =>
        n === 1 ? TOOL_CALL_ECHO : JSON.stringify({ type: "final", message: "完成了" }),
    );

    await run({ model, maxSteps: 10, events });

    assert.equal(countType(events, "run_start"), 1, "run_start 应只由循环入口发出一次");
    assert.equal(countType(events, "assistant"), 1);

    const ends = runEnds(events);
    assert.equal(ends.length, 1);
    assert.equal(ends[0].reason, "final");

    // 前端在 assistant 事件里固化占位消息、在 run_end 里复位状态，顺序不能反
    assert.ok(
        events.findIndex((e) => e.type === "assistant") <
            events.findIndex((e) => e.type === "run_end"),
        "assistant 必须先于 run_end",
    );
});

test("超出步数上限：只发 run_end(max_steps_exceeded)，没有 assistant，并抛错", async () => {
    const events: AgentEvent[] = [];
    const model = wrapModel(() => TOOL_CALL_ECHO);

    await assert.rejects(run({ model, maxSteps: 3, events }), /maximum number of steps/);

    assert.equal(countType(events, "tool_end"), 3, "三步工具都执行了");
    assert.equal(countType(events, "assistant"), 0, "不会有最终回答——前端必须自己收尾");

    const ends = runEnds(events);
    assert.equal(ends.length, 1);
    assert.equal(ends[0].reason, "max_steps_exceeded");
    assert.equal(ends[0].step, 3);
});

test("解析失败是可恢复错误：每步发一条 run_error 后继续，最终仍走到 run_end", async () => {
    const events: AgentEvent[] = [];
    const model = wrapModel(() => "这不是 JSON");

    await assert.rejects(run({ model, maxSteps: 3, events }), /maximum number of steps/);

    const parseErrors = runErrors(events).filter(
        (e) => e.stage === "parse_agent_response",
    );
    assert.equal(parseErrors.length, 3, "每步各一条解析错误");

    // 循环没有因为解析失败而中断：每步都真的又调了一次模型
    assert.equal(countType(events, "model_raw"), 3);

    assert.equal(runEnds(events).length, 1);
    assert.equal(runEnds(events)[0].reason, "max_steps_exceeded");
});

test("模型调用失败是致命错误：发 run_error(model_generate) 后抛出，不再发 run_end", async () => {
    const events: AgentEvent[] = [];
    const model = wrapModel((n) => {
        if (n === 1) return TOOL_CALL_ECHO;
        throw new Error("模型服务超时");
    });

    await assert.rejects(run({ model, maxSteps: 10, events }), /模型服务超时/);

    const fatal = runErrors(events).filter((e) => e.stage === "model_generate");
    assert.equal(fatal.length, 1);
    assert.equal(fatal[0].step, 2);

    assert.equal(countType(events, "assistant"), 0);
    assert.equal(runEnds(events).length, 0, "致命路径不发 run_end，前端只能靠 run_error 收尾");
});

test("循环外的阶段异常不带任何终止事件，Web 层必须自行补发 run_error", async () => {
    const events: AgentEvent[] = [];
    const model = wrapModel(() => TOOL_CALL_ECHO);

    // 落盘回调抛错，模拟历史压缩 / 会话持久化 / 技能规划这一类循环外异常
    await assert.rejects(
        runLocalAgentLoop({
            userInput: "测试任务",
            modelClient: model,
            tools: new Map([["echo", ECHO_TOOL]]),
            eventBus: makeBus(events),
            maxSteps: 3,
            previousMessages: [],
            depth: 0,
            onMessagesUpdated: async () => {
                throw new Error("磁盘写入失败");
            },
        }),
        /磁盘写入失败/,
    );

    assert.equal(runEnds(events).length, 0);
    assert.equal(runErrors(events).length, 0);
    assert.equal(
        events.some((e) => e.type === "run_end" || e.type === "run_error"),
        false,
        "前端收不到任何终止信号，若服务端不补发就会永远停在 isProcessing",
    );
});
