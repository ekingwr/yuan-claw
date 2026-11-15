/**
 * 外置任务计划（update_plan）离线测试
 * ============================================================
 * 不依赖真实 LLM / 网络：用脚本化的假 modelClient 精确编排每一步决策。
 *
 * 覆盖两层：
 *   - 单元层：update_plan 的参数校验、全量替换、in_progress 归一化、截断、事件
 *   - 集成层：计划清单被注入到送给模型的上下文末尾，且不落盘、不下传给子智能体
 *
 * 运行：
 *   node --import tsx --test test/plan.test.ts
 * ============================================================
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
    buildPlanReminder,
    createPlanState,
    createPlanTool,
    PLAN_TOOL_NAME,
} from "../src/agent/plan.js";
import { runLocalAgentLoop } from "../src/agent/run-local-agent-loop.js";
import type { AgentEvent, EventBus } from "../src/events/event-bus.js";
import type { Tool, ToolResult } from "../src/tools/types.js";
import type { ChatMessage } from "../src/memory/types.js";

// 与 plan.ts 内部常量保持一致（未导出，改动时需同步）
const MAX_PLAN_ITEMS = 20;
const MAX_ITEM_CHARS = 200;

const PLAN_REMINDER_MARKER = "以下是你为当前任务制定的计划";

function makeBus(events: AgentEvent[]): EventBus {
    return {
        emit: (e) => events.push(e),
        showProcessing() {},
        hideProcessing() {},
    };
}

function textOf(message: ChatMessage): string {
    return typeof message.content === "string"
        ? message.content
        : message.content.map((p) => (p.type === "text" ? p.text : "[图片]")).join("");
}

function todo(content: string, status?: string) {
    return status ? { content, status } : { content };
}

type PlanUpdateEvent = Extract<AgentEvent, { type: "plan_update" }>;

function planEvents(events: AgentEvent[]): PlanUpdateEvent[] {
    return events.filter((e): e is PlanUpdateEvent => e.type === "plan_update");
}

// ─── 单元层：update_plan 工具本身 ────────────────────────────────

test("update_plan：缺少或空的 todos 直接失败，不改动状态", async () => {
    const state = createPlanState();
    const tool = createPlanTool(state, makeBus([]));

    const missing = await tool.execute({});
    assert.equal(missing.success, false);

    const empty = await tool.execute({ todos: [] });
    assert.equal(empty.success, false);

    const notArray = await tool.execute({ todos: "1. 做事" });
    assert.equal(notArray.success, false);

    assert.deepEqual(state.items, []);
});

test("update_plan：任一项缺少 content 则整体拒绝，避免静默丢弃", async () => {
    const state = createPlanState();
    const tool = createPlanTool(state, makeBus([]));

    const result = await tool.execute({
        todos: [todo("第一步"), { status: "pending" }, todo("第三步")],
    });

    assert.equal(result.success, false);
    assert.match(String((result as { error: string }).error), /todos\[1\]/);
    assert.deepEqual(state.items, [], "失败时不应写入部分计划");
});

test("update_plan：成功时全量替换状态并 emit plan_update", async () => {
    const state = createPlanState();
    const events: AgentEvent[] = [];
    const tool = createPlanTool(state, makeBus(events));

    const first = await tool.execute({
        todos: [todo("调研项目结构", "in_progress"), todo("生成大纲"), todo("产出 pptx")],
    });

    assert.equal(first.success, true);
    assert.deepEqual(state.items.map((i) => i.status), [
        "in_progress",
        "pending",
        "pending",
    ]);

    const output = (first as { success: true; output: Record<string, unknown> }).output;
    assert.equal(output.total, 3);
    assert.equal(output.in_progress, 1);
    assert.equal(output.pending, 2);
    assert.equal(output.completed, 0);

    const emitted = planEvents(events);
    assert.equal(emitted.length, 1);
    assert.equal(emitted[0].items.length, 3);

    // 全量替换：第二次调用完全覆盖第一次
    await tool.execute({ todos: [todo("唯一一项", "completed")] });
    assert.equal(state.items.length, 1);
    assert.equal(state.items[0].content, "唯一一项");
    assert.equal(state.items[0].status, "completed");
    assert.equal(planEvents(events).length, 2);
});

test("update_plan：多项 in_progress 只保留第一个，其余降级并给出 note", async () => {
    const state = createPlanState();
    const tool = createPlanTool(state, makeBus([]));

    const result = await tool.execute({
        todos: [todo("A", "in_progress"), todo("B", "in_progress"), todo("C", "in_progress")],
    });

    assert.equal(result.success, true);
    assert.deepEqual(state.items.map((i) => i.status), [
        "in_progress",
        "pending",
        "pending",
    ]);

    const output = (result as { success: true; output: Record<string, unknown> }).output;
    assert.equal(output.in_progress, 1);
    assert.match(String(output.note), /只允许一项 in_progress/);
});

test("update_plan：未知 status 归一化为 pending，超出上限的项数与长度被截断", async () => {
    const state = createPlanState();
    const tool = createPlanTool(state, makeBus([]));

    const result = await tool.execute({
        todos: Array.from({ length: MAX_PLAN_ITEMS + 5 }, (_, i) =>
            todo(`第 ${i} 项 ${"y".repeat(MAX_ITEM_CHARS + 50)}`, "doing"),
        ),
    });

    assert.equal(result.success, true);
    assert.equal(state.items.length, MAX_PLAN_ITEMS);
    assert.ok(
        state.items.every((i) => i.status === "pending"),
        "未知 status 应回落为 pending",
    );
    assert.ok(
        state.items.every((i) => i.content.length <= MAX_ITEM_CHARS + 1),
        "单项描述应被截断（+1 为省略号）",
    );

    const output = (result as { success: true; output: Record<string, unknown> }).output;
    assert.match(String(output.note), /超过上限/);
});

test("buildPlanReminder：计划为空返回 null，非空则渲染清单与进度", () => {
    const empty = createPlanState();
    assert.equal(buildPlanReminder(empty), null);

    const state = createPlanState();
    state.items = [
        { content: "调研", status: "completed" },
        { content: "编写", status: "in_progress" },
        { content: "验证", status: "pending" },
    ];

    const reminder = buildPlanReminder(state);
    assert.ok(reminder);
    assert.equal(reminder.role, "user");

    const text = textOf(reminder);
    assert.ok(text.includes(PLAN_REMINDER_MARKER));
    assert.ok(text.includes("[x] 1. 调研"));
    assert.ok(text.includes("[~] 2. 编写"));
    assert.ok(text.includes("[ ] 3. 验证"));
    assert.ok(text.includes("进度：1/3 项已完成"));
});

// ─── 集成层：计划在 Agent 循环里的注入行为 ────────────────────────

const ECHO_TOOL: Tool = {
    name: "echo",
    description: "fake echo",
    riskLevel: "safe",
    async execute(): Promise<ToolResult> {
        return { success: true, output: "ok" };
    },
};

/** 按 system prompt 区分规划调用与协调者调用 */
function isPlannerCall(messages: ChatMessage[]): boolean {
    return textOf(messages[0]).includes("任务规划助手");
}

test("协调者：计划清单注入上下文末尾、不落盘，且子智能体拿不到 update_plan", async () => {
    const seen: ChatMessage[][] = [];
    const persisted: ChatMessage[][] = [];

    let coordinatorStep = 0;
    const modelClient = {
        async generate(messages: ChatMessage[]): Promise<string> {
            if (isPlannerCall(messages)) {
                return JSON.stringify({
                    taskAnalysis: "t",
                    selectedSkills: [],
                    needsTools: true,
                    plan: "p",
                });
            }

            seen.push(messages.map((m) => ({ ...m })));
            coordinatorStep += 1;

            if (coordinatorStep === 1) {
                return JSON.stringify({
                    type: "tool_call",
                    toolName: PLAN_TOOL_NAME,
                    args: {
                        todos: [
                            todo("调研 tax-simulator 项目结构", "in_progress"),
                            todo("编写 PPT 大纲"),
                            todo("生成 pptx 文件"),
                        ],
                    },
                });
            }
            if (coordinatorStep === 2) {
                return JSON.stringify({
                    type: "tool_call",
                    toolName: "echo",
                    args: {},
                });
            }
            return JSON.stringify({ type: "final", message: "done" });
        },
    };

    await runLocalAgentLoop({
        userInput: "编写一个介绍 tax-simulator 项目的 PPT",
        modelClient,
        tools: new Map([["echo", ECHO_TOOL]]),
        eventBus: makeBus([]),
        maxSteps: 10,
        previousMessages: [],
        depth: 0,
        onMessagesUpdated: async (m) => {
            persisted.push(m.map((x) => ({ ...x })));
        },
    });

    assert.equal(seen.length, 3, "协调者应经历 3 次模型调用");

    // update_plan 出现在协调者的系统提示词工具清单里
    assert.ok(textOf(seen[0][0]).includes(PLAN_TOOL_NAME));

    // 第 1 步：计划尚未建立，不应有计划提醒
    assert.ok(!seen[0].some((m) => textOf(m).includes(PLAN_REMINDER_MARKER)));

    // 第 2、3 步：计划提醒钉在上下文末尾
    for (const call of [seen[1], seen[2]]) {
        const last = textOf(call[call.length - 1]);
        assert.ok(last.includes(PLAN_REMINDER_MARKER), "计划提醒应在最后一条");
        assert.ok(last.includes("调研 tax-simulator 项目结构"));
        assert.ok(last.includes("进度：0/3 项已完成"));
    }

    // 计划提醒是瞬时的：任何一次持久化快照里都不该出现
    assert.ok(
        persisted.every((snap) => !snap.some((m) => textOf(m).includes(PLAN_REMINDER_MARKER))),
        "计划提醒不应写入 session 记忆",
    );
});

test("子智能体（depth=1）的系统提示词里没有 update_plan", async () => {
    const seen: ChatMessage[][] = [];

    await runLocalAgentLoop({
        userInput: "调研一下这个项目的依赖",
        modelClient: {
            async generate(messages: ChatMessage[]): Promise<string> {
                seen.push(messages.map((m) => ({ ...m })));
                return JSON.stringify({ type: "final", message: "结论" });
            },
        },
        tools: new Map([["echo", ECHO_TOOL]]),
        eventBus: makeBus([]),
        maxSteps: 5,
        previousMessages: [],
        depth: 1,
        agentRole: "代码检索专家",
        enableSubagents: false,
    });

    assert.ok(!textOf(seen[0][0]).includes(PLAN_TOOL_NAME));
});
