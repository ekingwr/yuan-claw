import type { ChatMessage } from "../memory/types.js";
import type { EventBus, PlanItem, PlanItemStatus } from "../events/event-bus.js";
import { formatPlanItems } from "../events/event-bus.js";
import type { Tool, ToolResult } from "../tools/types.js";

/**
 * 外置任务计划（todo）
 * ============================================================
 * ReAct 把「接下来做什么」隐含在对话历史里，历史一被压缩或被大量工具输出淹没，
 * 计划就跟着消失，模型只能靠最近的几条消息猜下一步——这是长任务跑偏的主因之一。
 *
 * 对策是把计划外置成一份独立于对话历史的结构化状态：
 *   - 模型通过 update_plan 工具显式建立/更新计划；
 *   - 每一步都把计划渲染后钉在上下文末尾（注意力最强的位置）；
 *   - 计划体积恒定（几行文本），不参与历史压缩，也不会随工具输出膨胀。
 *
 * 这是「轻量 Plan-and-Execute」：只在需要时规划，执行仍是 ReAct，
 * 因此保留了 ReAct 的逐步适应能力，又拿到了计划的可见性与不跑偏。
 *
 * 状态只存在于单次 runLocalAgentLoop 调用内（一个用户任务），不写入 messages、
 * 不持久化到 session：计划是执行期的脚手架，任务结束即失去意义，
 * 跨轮次残留的计划反而会误导下一次对话。UI 通过 plan_update 事件实时观察。
 * ============================================================
 */

export const PLAN_TOOL_NAME = "update_plan";

/** 计划项数量上限：再多就说明任务该拆成多次对话或用子智能体委派了。 */
const MAX_PLAN_ITEMS = 20;

/** 单项描述长度上限，防止模型把整段分析塞进一条 todo。 */
const MAX_ITEM_CHARS = 200;

const VALID_STATUSES: ReadonlySet<string> = new Set<PlanItemStatus>([
    "pending",
    "in_progress",
    "completed",
]);

/**
 * 计划状态容器。用对象包一层而不是裸数组，是为了让工具与循环共享同一个可变引用：
 * 工具负责改，循环每步负责读。
 */
export type PlanState = {
    items: PlanItem[];
};

export function createPlanState(): PlanState {
    return { items: [] };
}

function normalizeStatus(raw: unknown): PlanItemStatus {
    return typeof raw === "string" && VALID_STATUSES.has(raw)
        ? (raw as PlanItemStatus)
        : "pending";
}

/**
 * 同时只允许一项 in_progress——多项并行会让「当前该做什么」重新变得含糊，
 * 等于放弃了外置计划的意义。保留第一个，其余降级为 pending。
 */
function enforceSingleInProgress(items: PlanItem[]): boolean {
    let seenInProgress = false;
    let demoted = false;

    for (const item of items) {
        if (item.status !== "in_progress") {
            continue;
        }
        if (seenInProgress) {
            item.status = "pending";
            demoted = true;
        } else {
            seenInProgress = true;
        }
    }

    return demoted;
}

/** 渲染成紧凑的清单文本，供注入上下文与 CLI 展示共用。 */
export function renderPlanSection(items: PlanItem[]): string {
    const completed = items.filter((item) => item.status === "completed").length;

    return [
        formatPlanItems(items),
        "",
        `进度：${completed}/${items.length} 项已完成`,
    ].join("\n");
}

/**
 * 构建注入上下文末尾的计划提醒。计划为空时返回 null（短任务无需规划，
 * 不应该硬塞一段噪音进上下文）。
 */
export function buildPlanReminder(state: PlanState): ChatMessage | null {
    if (state.items.length === 0) {
        return null;
    }

    return {
        role: "user",
        content: [
            "以下是你为当前任务制定的计划：",
            renderPlanSection(state.items),
            "",
            "请继续推进标记为 [~] 的那一项；每完成一项就调用 update_plan 更新状态。",
            "不要去做计划之外、用户没有要求的事情；若确实需要调整路线，先用 update_plan 改写计划再执行。",
        ].join("\n"),
    };
}

/**
 * ===== update_plan：建立或全量替换任务计划 =====
 *
 * 采用全量替换而非增量打补丁：基于下标的 patch 极易与模型的计数错位，
 * 而整表重写每次只有几行文本，成本可以忽略，且天然幂等。
 */
export function createPlanTool(state: PlanState, eventBus: EventBus): Tool {
    return {
        name: PLAN_TOOL_NAME,
        // 只改循环内的计划状态，无外部副作用，故为 safe、无需确认。
        riskLevel: "safe",
        description:
            "建立或更新当前任务的执行计划（todo 清单）。计划会在后续每一步都展示给你，" +
            "用于在长任务中保持方向、不遗漏步骤。" +
            "调用方式为全量替换：每次都要传入完整的清单，而不是只传变化的那几项。" +
            "参数: todos (array, 必填): 每个元素为 {content: string(必填,一句话描述该步骤), " +
            'status: "pending"|"in_progress"|"completed"(可选,缺省为 pending)}；' +
            `最多 ${MAX_PLAN_ITEMS} 项，同一时刻只应有一项为 in_progress。`,
        async execute(rawArgs: Record<string, unknown>): Promise<ToolResult> {
            const args = rawArgs as { todos?: unknown };

            if (!Array.isArray(args.todos) || args.todos.length === 0) {
                return {
                    success: false,
                    error: `update_plan 需要非空数组参数 "todos"，每个元素形如 {content, status?}。`,
                };
            }

            // 任一项缺少 content 就整体拒绝：静默丢弃会让模型误以为计划已按它想的样子生效。
            const items: PlanItem[] = [];
            for (let i = 0; i < args.todos.length; i += 1) {
                const record = (args.todos[i] ?? {}) as {
                    content?: unknown;
                    status?: unknown;
                };
                const content =
                    typeof record.content === "string" ? record.content.trim() : "";

                if (!content) {
                    return {
                        success: false,
                        error: `todos[${i}] 缺少非空字符串字段 "content"。请重新提交完整的计划。`,
                    };
                }

                items.push({
                    content:
                        content.length > MAX_ITEM_CHARS
                            ? `${content.slice(0, MAX_ITEM_CHARS)}…`
                            : content,
                    status: normalizeStatus(record.status),
                });
            }

            const truncated = items.length > MAX_PLAN_ITEMS;
            if (truncated) {
                items.length = MAX_PLAN_ITEMS;
            }

            const demoted = enforceSingleInProgress(items);

            state.items = items;

            eventBus.emit({ type: "plan_update", items: [...items] });

            const completed = items.filter((i) => i.status === "completed").length;
            const inProgress = items.filter((i) => i.status === "in_progress").length;

            const notes: string[] = [];
            if (truncated) {
                notes.push(`计划项超过上限 ${MAX_PLAN_ITEMS}，已截断。`);
            }
            if (demoted) {
                notes.push("同一时刻只允许一项 in_progress，多余项已降级为 pending。");
            }

            return {
                success: true,
                output: {
                    total: items.length,
                    completed,
                    in_progress: inProgress,
                    pending: items.length - completed - inProgress,
                    ...(notes.length > 0 ? { note: notes.join(" ") } : {}),
                },
            };
        },
    };
}
