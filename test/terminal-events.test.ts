/**
 * 终止信号判定与去重包装。
 *
 * Web 层靠它决定「循环抛错后要不要补发 run_error」：已经发过终止信号的不补，
 * 否则模型报错会弹两次提示；什么都没发的必须补，否则前端一直转圈。
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { isTerminalEvent, trackTerminalEvents } from "../src/events/terminal-events.js";
import type { AgentEvent, EventBus } from "../src/events/event-bus.js";

type RunErrorStage = Extract<AgentEvent, { type: "run_error" }>["stage"];

const runError = (stage: RunErrorStage): AgentEvent => ({
    type: "run_error",
    step: 1,
    stage,
    error: "boom",
});

function makeRecorder() {
    const seen: AgentEvent[] = [];
    const calls: string[] = [];
    const bus: EventBus = {
        emit: (e) => seen.push(e),
        showProcessing: () => calls.push("show"),
        hideProcessing: () => calls.push("hide"),
    };
    return { bus, seen, calls };
}

test("isTerminalEvent：run_end 与致命 run_error 算终止，可恢复的解析错误不算", () => {
    assert.equal(isTerminalEvent({ type: "run_end", reason: "final", step: 3 }), true);
    assert.equal(
        isTerminalEvent({ type: "run_end", reason: "max_steps_exceeded", step: 3 }),
        true,
        "超步数也是终止：循环随后就抛出，前端靠这条事件收尾",
    );

    assert.equal(isTerminalEvent(runError("model_generate")), true);
    assert.equal(isTerminalEvent(runError("confirmation")), true);
    assert.equal(isTerminalEvent(runError("server")), true);

    assert.equal(
        isTerminalEvent(runError("parse_agent_response")),
        false,
        "解析失败后循环会 continue 继续跑，不是终止信号",
    );
});

test("isTerminalEvent：中间过程事件都不算终止", () => {
    const nonTerminal: AgentEvent[] = [
        { type: "run_start", input: "x" },
        { type: "model_raw", text: "{}", step: 1 },
        { type: "tool_start", toolName: "echo", args: {}, step: 1 },
        { type: "tool_end", toolName: "echo", success: false, result: "err", step: 1 },
        { type: "tool_error", toolName: "echo", error: "err", step: 1 },
        { type: "plan_update", items: [] },
        { type: "assistant", message: "done" },
        { type: "streaming_token", text: "a", done: true },
        { type: "subagent_end", agent: "a", success: false, error: "e" },
    ];

    for (const event of nonTerminal) {
        assert.equal(isTerminalEvent(event), false, `${event.type} 不应被判定为终止`);
    }
});

test("trackTerminalEvents：原样转发事件，不改动内容与顺序", () => {
    const { bus, seen } = makeRecorder();
    const tracked = trackTerminalEvents(bus);

    const events: AgentEvent[] = [
        { type: "run_start", input: "x" },
        runError("parse_agent_response"),
        { type: "tool_start", toolName: "echo", args: {}, step: 2 },
        { type: "assistant", message: "done" },
        { type: "run_end", reason: "final", step: 2 },
    ];
    for (const event of events) {
        tracked.bus.emit(event);
    }

    assert.deepEqual(seen, events);
});

test("trackTerminalEvents：hasTerminated 只在收到终止信号后翻正，且不可逆", () => {
    const { bus } = makeRecorder();
    const tracked = trackTerminalEvents(bus);

    assert.equal(tracked.hasTerminated(), false, "初始未终止");

    tracked.bus.emit({ type: "run_start", input: "x" });
    assert.equal(tracked.hasTerminated(), false);

    // 可恢复错误：循环还在跑，不能当成终止
    tracked.bus.emit(runError("parse_agent_response"));
    tracked.bus.emit(runError("parse_agent_response"));
    assert.equal(tracked.hasTerminated(), false, "解析失败不应置为已终止");

    tracked.bus.emit(runError("model_generate"));
    assert.equal(tracked.hasTerminated(), true);

    // 之后再来什么都保持 true，避免补发逻辑反复触发
    tracked.bus.emit({ type: "run_start", input: "y" });
    assert.equal(tracked.hasTerminated(), true);
});

test("trackTerminalEvents：showProcessing / hideProcessing 透传到内层总线", () => {
    const { bus, calls } = makeRecorder();
    const tracked = trackTerminalEvents(bus);

    tracked.bus.showProcessing();
    tracked.bus.hideProcessing();

    assert.deepEqual(calls, ["show", "hide"]);
});
