import type { ChatMessage } from "../memory/types.js";
import { countMessagesChars } from "../memory/message-chars.js";

type ModelClient = {
    generate: (messages: ChatMessage[]) => Promise<string>;
};

/** 历史对话压缩阈值：超过此字符数时触发 LLM 压缩 */
export const COMPRESSION_THRESHOLD = 1_000_000;

/**
 * 压缩后保留最近的消息数量（不压缩，保持原文）。
 *
 * ReAct 循环每一步产生 2 条消息（assistant 的 tool_call + user 的工具结果），
 * 因此这个值除以 2 才是模型能原文回看的「步数」。取值 20 即保留最近约 10 步，
 * 让模型在压缩发生后仍有足够的近期细节做判断，而不是只看到 3 步就开始跑偏。
 *
 * 上界受 MAX_COMPRESS_INPUT 约束：单条工具结果最多 12000 字符，
 * 20 条约 240k 字符，剩余的 760k 仍落在 800k 的压缩输入预算内。
 */
const PRESERVE_RECENT_COUNT = 20;

/**
 * 送入压缩模型的最大字符数，超出部分取头尾各一半。
 * 与 COMPRESSION_THRESHOLD 保持 0.8 倍关系：刚过阈值的历史可以完整压缩，
 * 只有极度膨胀的历史才会丢弃中段。
 */
const MAX_COMPRESS_INPUT = 800_000;

/** 将消息序列化为文本格式供 LLM 压缩 */
function serializeMessages(messages: ChatMessage[]): string {
    const lines: string[] = [];
    for (const msg of messages) {
        const content =
            typeof msg.content === "string"
                ? msg.content
                : msg.content
                      .map((p) => (p.type === "text" ? p.text : "[图片]"))
                      .join("");
        lines.push(`[${msg.role}]: ${content}`);
    }
    return lines.join("\n");
}

const COMPRESSION_SYSTEM_PROMPT = `你是一个对话历史压缩助手。你的任务是将一段很长的多轮对话历史压缩为一个结构化的摘要。

要求：
1. 保留所有关键信息：用户的核心问题、重要的决策、执行过的操作及其结果、文件路径、代码片段的关键逻辑
2. 按时间顺序组织摘要
3. 使用以下固定格式输出：

## 对话摘要

### 用户目标
[用户想要完成的核心任务]

### 关键上下文
- [重要的背景信息、文件路径、配置等]

### 已完成的操作
1. [操作1]: [结果/结论]
2. [操作2]: [结果/结论]
...

### 重要结论与决策
- [决策1及原因]
- [决策2及原因]

### 当前状态
[对话截止时的状态：正在做什么、下一步计划等]

注意：
- 「用户目标」必须忠实还原用户最初的原话诉求，不要改写成执行过程中衍生出的子目标
- 不要遗漏任何工具调用的关键结果（如文件内容、搜索结果的重要部分）
- 代码相关的对话要保留关键代码逻辑的描述
- 压缩后的摘要应该足够详细，使得后续的 LLM 能够基于此摘要继续对话而不丢失上下文`;

/**
 * 压缩历史对话（无条件执行）。
 *
 * 将较早的消息通过 LLM 压缩为结构化摘要，保留最近 N 条消息原文不变。
 * 返回压缩后的消息数组：
 *   [...system 消息, 首条消息原文, 压缩摘要(user角色), ...最近N条原文消息]
 *
 * 首条消息（通常是对话的起点）被单独钉住、永不参与压缩：多次压缩会让摘要
 * 退化成「摘要的摘要」，把最初的需求逐轮稀释掉，钉住原文可以杜绝这条路径。
 *
 * 由 compressHistoryIfNeeded（自动触发）与 Web 端「手动压缩」按钮共用。
 */
export async function compressHistory(
    messages: ChatMessage[],
    modelClient: ModelClient,
): Promise<ChatMessage[]> {
    const totalChars = countMessagesChars(messages);

    // 分离 system 消息（不参与压缩）
    const systemMessages = messages.filter((m) => m.role === "system");
    const nonSystemMessages = messages.filter((m) => m.role !== "system");

    if (nonSystemMessages.length <= PRESERVE_RECENT_COUNT + 1) {
        // 除去钉住的首条后，剩余消息不足以再切出「待压缩」段，直接返回
        return messages;
    }

    // 首条钉住；其余再分为需要压缩的旧消息和保留原文的最近消息
    const [pinnedFirstMessage, ...restMessages] = nonSystemMessages;
    const olderMessages = restMessages.slice(0, -PRESERVE_RECENT_COUNT);
    const recentMessages = restMessages.slice(-PRESERVE_RECENT_COUNT);

    const serializedHistory = serializeMessages(olderMessages);

    // 如果序列化后的文本过长，截取前后部分（保留头尾信息）
    let textToCompress = serializedHistory;
    if (textToCompress.length > MAX_COMPRESS_INPUT) {
        const half = Math.floor(MAX_COMPRESS_INPUT / 2);
        const head = textToCompress.slice(0, half);
        const tail = textToCompress.slice(-half);
        textToCompress = `${head}\n\n...[中间部分省略]...\n\n${tail}`;
    }

    try {
        const compressionResult = await modelClient.generate([
            { role: "system", content: COMPRESSION_SYSTEM_PROMPT },
            {
                role: "user",
                content: `请压缩以下对话历史：\n\n${textToCompress}`,
            },
        ]);

        // 构建压缩后的摘要消息
        const summaryMessage: ChatMessage = {
            role: "user",
            content: `[以下是之前对话的压缩摘要，共 ${olderMessages.length} 条消息，原始约 ${totalChars} 字符]\n\n${compressionResult}`,
        };

        // 返回：system 消息 + 钉住的首条原文 + 压缩摘要 + 最近消息
        return [
            ...systemMessages,
            pinnedFirstMessage,
            summaryMessage,
            ...recentMessages,
        ];
    } catch (error) {
        // 压缩失败时降级：仅保留钉住的首条与最近的消息（不做 LLM 压缩）
        console.warn(
            "[compress-history] LLM compression failed, falling back to recent messages only:",
            error instanceof Error ? error.message : String(error),
        );
        return [...systemMessages, pinnedFirstMessage, ...recentMessages];
    }
}

/**
 * 压缩历史对话。
 *
 * 当消息总字符数超过阈值（COMPRESSION_THRESHOLD）时，调用 compressHistory 压缩；
 * 未超过阈值则原样返回，不产生 LLM 调用。
 */
export async function compressHistoryIfNeeded(
    messages: ChatMessage[],
    modelClient: ModelClient,
): Promise<ChatMessage[]> {
    if (countMessagesChars(messages) <= COMPRESSION_THRESHOLD) {
        return messages;
    }
    return compressHistory(messages, modelClient);
}
