import Foundation
import Testing
@testable import Rapid

/// The anti-confabulation guidance used to be prepended to the system row on
/// the round that carried a tool result and stripped again on the next turn.
/// The system row is the head of every prompt and the engine's prefix cache
/// reuses a stored prompt only up to the first differing token, so one web
/// search cost the whole conversation two cold prefills (0.14.1, Qwen3.8-27B,
/// 5.7k-token conversation: ~20 s each against 1.5 s for an append-only turn).
///
/// These tests pin where the guidance lives now — the newest user row's
/// wire-only trailer — and the property that placement buys: every row
/// before the newest user message, the system row first of all, is
/// byte-identical between the tool round and the rounds around it.
@Suite("Tool guidance rides the newest user row")
struct ToolGuidanceTrailerTests {

    private static let date = "[CURRENT DATE]\nToday is Friday, 12 September 2026."

    private static func assemble(_ body: [ChatMessage], toolsAdvertised: Bool = true) -> [ChatMessage] {
        var history = ChatViewModel.addingInstructionLayers(
            to: body,
            dateContext: date,
            global: "Answer briefly.",
            conversation: ""
        )
        history = ChatViewModel.stampingClockContext(on: history)
        return ChatViewModel.stampingToolGuidance(on: history, toolsAdvertised: toolsAdvertised)
    }

    private static let asked = Date(timeIntervalSince1970: 1_789_300_000)

    private static let roundOne: [ChatMessage] = {
        [
            ChatMessage(role: .user, content: "Name three colors.", createdAt: asked),
            ChatMessage(role: .assistant, content: "Red, blue, and green.", createdAt: asked),
            ChatMessage(role: .user, content: "weather in Tokyo?", createdAt: asked),
        ]
    }()

    private static let roundTwo: [ChatMessage] = {
        roundOne + [
            ChatMessage(
                role: .assistant,
                toolCalls: [ToolCall(id: "w1", name: "weather", arguments: "{\"city\":\"Tokyo\"}")],
                createdAt: asked
            ),
            ChatMessage(role: .tool, content: "{\"temp_c\": 29.2}", toolCallID: "w1", createdAt: asked),
        ]
    }()

    private static let nextTurn: [ChatMessage] = {
        roundTwo + [
            ChatMessage(role: .assistant, content: "It's 29.2°C in Tokyo.", createdAt: asked),
            ChatMessage(role: .user, content: "what is the capital of France?", createdAt: asked),
        ]
    }()

    @Test("The guidance is a trailer on the newest user row, behind the clock, never in the system row")
    func guidanceLandsOnTheNewestUserRow() {
        let wire = Self.assemble(Self.roundTwo)
        let system = wire[0]
        #expect(system.role == .system)
        #expect(!system.content.contains(ChatViewModel.toolGuidance))

        let newestUser = wire[3]
        #expect(newestUser.role == .user)
        #expect(newestUser.content == "weather in Tokyo?", "the transcript row stays prose-only")
        let suffix = newestUser.wireSuffix ?? ""
        #expect(suffix.hasPrefix("[MESSAGE SENT]"), "the clock trailer still comes first")
        #expect(suffix.hasSuffix(ChatViewModel.toolGuidance))
        #expect(newestUser.modelContent.hasSuffix(ChatViewModel.toolGuidance))

        // No other row carries it: not the earlier user row, not the tool row.
        #expect(wire[1].wireSuffix?.contains(ChatViewModel.toolGuidance) == false)
        #expect(wire[5].role == .tool)
        #expect(wire[5].wireSuffix == nil)
    }

    @Test("The rows before the newest user message are byte-identical across the tool round")
    func headOfThePromptIsStableAcrossTheToolRound() {
        let before = Self.assemble(Self.roundOne)
        let during = Self.assemble(Self.roundTwo)
        let after = Self.assemble(Self.nextTurn)

        // Round one (no tool result yet) and round two (tool result in play)
        // differ ONLY from the newest user row on; the system row and the
        // earlier turn are the same bytes on the wire.
        #expect(during[0].content == before[0].content)
        #expect(during[1] == before[1])
        #expect(during[2] == before[2])
        #expect(during[3].modelContent != before[3].modelContent)
        #expect(during[3].content == before[3].content)

        // The next turn no longer carries a tool result, so the guidance is
        // gone again — and it leaves from the user row it rode, not from the
        // head of the prompt. Everything before that row is untouched.
        #expect(after[0].content == during[0].content)
        #expect(after[1] == during[1])
        #expect(after[2] == during[2])
        #expect(after[3].wireSuffix?.contains(ChatViewModel.toolGuidance) == false)
        #expect(after.last?.wireSuffix?.contains(ChatViewModel.toolGuidance) == false)
    }

    @Test("No guidance without advertised tools or without a tool result this turn (#1549)")
    func gateIsUnchanged() {
        let noTools = Self.assemble(Self.roundTwo, toolsAdvertised: false)
        #expect(!noTools.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
        let noResult = Self.assemble(Self.roundOne)
        #expect(!noResult.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
    }

    @Test("A trim that drops the tool result drops the guidance with it")
    func trimmedEvidenceTakesTheInstruction() {
        // The caller stamps the TRIMMED history. Model the trim that keeps the
        // current turn's user row but elides its tool result: the guidance
        // must not claim a tool result that is no longer on the wire.
        var trimmed = Self.roundTwo
        trimmed.removeLast(2)
        let wire = Self.assemble(trimmed)
        #expect(!wire.contains { $0.wireSuffix?.contains(ChatViewModel.toolGuidance) == true })
    }

    @Test("On the wire, round two's body extends round one's up to the newest user row")
    func wireBodySharesThePrefixUpToTheNewestUserRow() async throws {
        func request(_ messages: [ChatMessage]) -> ChatStreamClient.Request {
            ChatStreamClient.Request(alias: "test-model", messages: messages, tools: nil, supportsImageInput: false)
        }
        let oneData = try #require(await WireBodyCaptureProtocol.capture(request(Self.assemble(Self.roundOne))))
        let twoData = try #require(await WireBodyCaptureProtocol.capture(request(Self.assemble(Self.roundTwo))))
        let one = try #require(String(data: oneData, encoding: .utf8))
        let two = try #require(String(data: twoData, encoding: .utf8))
        // The shared prefix must reach past the system row and the first
        // exchange, into the newest user row's own object.
        let marker = "\"content\":\"weather in Tokyo?"
        let oneCut = try #require(one.range(of: marker)?.upperBound)
        let head = String(one[..<oneCut])
        #expect(two.hasPrefix(head),
                "the system row and every earlier row must serialize identically whether or not the round carries a tool result")
        #expect(two.contains(marker))
        #expect(!two.contains("\"content\":\"You have access to tools"),
                "the guidance must not become a system/user row of its own")
    }
}
