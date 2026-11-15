import type { AgentEvent, EventBus } from "./event-bus.js";

/**
 * 判断一个事件是否意味着「本轮已经结束」。
 *
 * run_error 的 parse_agent_response 阶段是**可恢复**错误：循环塞入纠正提示后会
 * continue 继续跑（见 run-local-agent-loop.ts 的解析失败分支），因此不算终止。
 * 其余 stage 在 emit 之后都会紧接着抛出，本轮到此为止。
 */
export function isTerminalEvent(event: AgentEvent): boolean {
    if (event.type === "run_end") return true;
    return event.type === "run_error" && event.stage !== "parse_agent_response";
}

/**
 * 包一层 EventBus，记录本轮是否已经发出过终止信号。
 *
 * 用途：Agent 循环里的致命错误会先 emit run_error 再抛出，但压缩、落盘、技能规划
 * 等阶段的异常不带任何事件。外层 catch 无从区分这两种情况——一律补发 run_error
 * 会让模型报错弹两次提示，完全不发又会让前端收不到任何信号、一直转圈。
 *
 * 包装只增加一个标记，不改变事件的转发内容与顺序。
 */
export function trackTerminalEvents(inner: EventBus): {
    bus: EventBus;
    hasTerminated: () => boolean;
} {
    let terminated = false;

    return {
        bus: {
            ...inner,
            emit(event: AgentEvent) {
                if (isTerminalEvent(event)) {
                    terminated = true;
                }
                inner.emit(event);
            },
        },
        hasTerminated: () => terminated,
    };
}
