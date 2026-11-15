/**
 * 历史对话压缩机制离线测试
 * ============================================================
 * 不依赖真实 LLM / 网络：用一个假的 modelClient 记录调用次数与收到的消息，
 * 确定性地验证压缩的触发条件与输出结构。
 *
 * 重点覆盖「防止长任务跑偏」的三条不变量：
 *   - 首条消息被钉住，永不参与压缩（阻断「摘要的摘要」稀释最初需求）
 *   - 最近 PRESERVE_RECENT_COUNT 条原文保留
 *   - system 消息不参与压缩
 *
 * 运行：
 *   node --import tsx --test test/compress-history.test.ts
 * ============================================================
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
    compressHistory,
    compressHistoryIfNeeded,
    COMPRESSION_THRESHOLD,
} from "../src/agent/compress-history.js";
import { countMessagesChars } from "../src/memory/message-chars.js";
import type { ChatMessage } from "../src/memory/types.js";

// 与 compress-history.ts 内部常量保持一致（该常量未导出，改动时需同步）
const PRESERVE_RECENT_COUNT = 20;

const SUMMARY = "## 对话摘要\n\n### 用户目标\n用户请求帮助编写排序算法。\n";

function createMockModelClient(response: string, shouldFail = false) {
    const received: ChatMessage[][] = [];
    return {
        async generate(messages: ChatMessage[]): Promise<string> {
            received.push(messages);
            if (shouldFail) {
                throw new Error("Mock: model unavailable");
            }
            return response;
        },
        get callCount() {
            return received.length;
        },
        /** 最近一次压缩调用中，送给压缩模型的历史正文 */
        get lastPayload(): string {
            const last = received[received.length - 1];
            return String(last?.[1]?.content ?? "");
        },
    };
}

/** 每条约 5000 字符，totalChars 决定条数 */
function generateLongHistory(totalChars: number, perMessage = 5000): ChatMessage[] {
    const count = Math.ceil(totalChars / perMessage);
    return Array.from({ length: count }, (_, i): ChatMessage => ({
        role: i % 2 === 0 ? "user" : "assistant",
        content: `[消息 #${i}] ${"x".repeat(perMessage)}`,
    }));
}

function textOf(message: ChatMessage): string {
    return typeof message.content === "string" ? message.content : "[multimodal]";
}

test("短对话不触发压缩，原样返回同一数组引用", async () => {
    const messages: ChatMessage[] = [
        { role: "user", content: "你好" },
        { role: "assistant", content: "你好！有什么可以帮助你的？" },
        { role: "user", content: "帮我写一个排序算法" },
        { role: "assistant", content: "好的，这是一个快速排序的实现..." },
    ];

    const mock = createMockModelClient(SUMMARY);
    const result = await compressHistoryIfNeeded(messages, mock);

    assert.equal(mock.callCount, 0);
    assert.equal(result, messages);
});

test("阈值边界：低于 COMPRESSION_THRESHOLD 不压缩，超过则压缩", async () => {
    const underThreshold = generateLongHistory(COMPRESSION_THRESHOLD * 0.9);
    assert.ok(countMessagesChars(underThreshold) < COMPRESSION_THRESHOLD);
    const mockUnder = createMockModelClient(SUMMARY);
    await compressHistoryIfNeeded(underThreshold, mockUnder);
    assert.equal(mockUnder.callCount, 0, "低于阈值不应调用 LLM");

    const overThreshold = generateLongHistory(COMPRESSION_THRESHOLD * 1.1);
    assert.ok(countMessagesChars(overThreshold) > COMPRESSION_THRESHOLD);
    const mockOver = createMockModelClient(SUMMARY);
    await compressHistoryIfNeeded(overThreshold, mockOver);
    assert.equal(mockOver.callCount, 1, "超过阈值应恰好调用一次 LLM");
});

test("压缩输出结构 = 钉住的首条 + 摘要 + 最近 20 条原文", async () => {
    const messages = generateLongHistory(COMPRESSION_THRESHOLD * 1.1);
    const originalFirst = messages[0];

    const mock = createMockModelClient(SUMMARY);
    const result = await compressHistory(messages, mock);

    assert.equal(mock.callCount, 1);
    assert.equal(result.length, 1 + 1 + PRESERVE_RECENT_COUNT);

    // 首条被钉住：内容与原对象完全一致，不是摘要的一部分
    assert.equal(textOf(result[0]), textOf(originalFirst));
    assert.match(textOf(result[1]), /以下是之前对话的压缩摘要/);
    assert.ok(textOf(result[1]).includes(SUMMARY), "摘要正文应嵌入其中");

    // 尾部保留的是最新的 20 条（而非最旧的）
    const tail = result.slice(2);
    assert.equal(tail.length, PRESERVE_RECENT_COUNT);
    assert.equal(textOf(tail[tail.length - 1]), textOf(messages[messages.length - 1]));
    assert.equal(textOf(tail[0]), textOf(messages[messages.length - PRESERVE_RECENT_COUNT]));

    assert.ok(
        countMessagesChars(result) < countMessagesChars(messages),
        "压缩后总字符数应显著下降",
    );
});

test("重复压缩时钉住的仍是同一条原始首条消息", async () => {
    const messages = generateLongHistory(COMPRESSION_THRESHOLD * 1.1);
    const originalFirst = messages[0];

    const first = await compressHistory(messages, createMockModelClient(SUMMARY));
    const second = await compressHistory(first, createMockModelClient(SUMMARY));
    const third = await compressHistory(second, createMockModelClient(SUMMARY));

    // 三轮压缩后起点仍未被稀释
    assert.equal(textOf(third[0]), textOf(originalFirst));
    assert.equal(third.length, 1 + 1 + PRESERVE_RECENT_COUNT);
});

test("system 消息不参与压缩，且仍排在最前", async () => {
    const messages: ChatMessage[] = [
        { role: "system", content: "你是一个有帮助的助手。" },
        ...generateLongHistory(COMPRESSION_THRESHOLD * 1.1),
    ];
    const originalFirst = messages[1];

    const mock = createMockModelClient(SUMMARY);
    const result = await compressHistory(messages, mock);

    assert.equal(result[0].role, "system");
    assert.equal(result[0].content, "你是一个有帮助的助手。");
    assert.equal(textOf(result[1]), textOf(originalFirst), "system 之后应是被钉住的首条");
    assert.match(textOf(result[2]), /以下是之前对话的压缩摘要/);
    assert.equal(result.length, 1 + 1 + 1 + PRESERVE_RECENT_COUNT);
});

test("压缩失败时降级：保留钉住的首条 + 最近 20 条，不抛异常", async () => {
    const messages = generateLongHistory(COMPRESSION_THRESHOLD * 1.1);
    const originalFirst = messages[0];

    const mock = createMockModelClient("", true);
    const result = await compressHistory(messages, mock);

    assert.equal(mock.callCount, 1);
    assert.equal(result.length, 1 + PRESERVE_RECENT_COUNT);
    assert.equal(textOf(result[0]), textOf(originalFirst));
    assert.equal(
        textOf(result[result.length - 1]),
        textOf(messages[messages.length - 1]),
    );
});

test("消息条数不足（<= 保留窗口 + 1）时直接返回，不发起 LLM 调用", async () => {
    const messages = generateLongHistory(
        (PRESERVE_RECENT_COUNT + 1) * 1000,
        1000,
    );
    assert.equal(messages.length, PRESERVE_RECENT_COUNT + 1);

    const mock = createMockModelClient(SUMMARY);
    const result = await compressHistory(messages, mock);

    assert.equal(mock.callCount, 0, "待压缩段为空时不应浪费一次 LLM 调用");
    assert.equal(result, messages);
});

test("待压缩段超过 MAX_COMPRESS_INPUT 时取头尾、省略中段", async () => {
    // 220 条 × 5000 字符 ≈ 1.1M，其中待压缩段约 199 条 ≈ 995k > 800k 上限
    const messages = generateLongHistory(COMPRESSION_THRESHOLD * 1.1);

    const mock = createMockModelClient(SUMMARY);
    await compressHistory(messages, mock);

    assert.ok(mock.lastPayload.includes("...[中间部分省略]..."));
    assert.ok(mock.lastPayload.includes("[消息 #1]"), "应保留头部");
    assert.ok(
        mock.lastPayload.includes(`[消息 #${messages.length - PRESERVE_RECENT_COUNT - 1}]`),
        "应保留尾部",
    );
});

test("多模态内容（图片）参与字符统计，按 500 字符估算", async () => {
    const messages: ChatMessage[] = [
        { role: "user", content: "看下这张图" },
        ...Array.from({ length: PRESERVE_RECENT_COUNT + 5 }, (_, i): ChatMessage => ({
            role: i % 2 === 0 ? "user" : "assistant",
            content: [
                { type: "text", text: `第 ${i} 步` },
                {
                    type: "image_url",
                    image_url: { url: "data:image/png;base64,AAAA" },
                },
            ],
        })),
    ];

    // 每条约 500+ 字符，总量仍远低于阈值 → 不压缩
    assert.ok(countMessagesChars(messages) < COMPRESSION_THRESHOLD);
    const mock = createMockModelClient(SUMMARY);
    const result = await compressHistoryIfNeeded(messages, mock);
    assert.equal(mock.callCount, 0);
    assert.equal(result, messages);
});
