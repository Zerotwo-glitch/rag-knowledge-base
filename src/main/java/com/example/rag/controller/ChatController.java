package com.example.rag.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.nio.charset.StandardCharsets;

/**
 * 多轮对话：流式输出 + 会话记忆。
 */
@RestController
public class ChatController {

    private static final MediaType UTF8_EVENT_STREAM =
            new MediaType("text", "event-stream", StandardCharsets.UTF_8);

    private final ChatMemory chatMemory;
    private final ChatClient chatClient;

    public ChatController(ChatMemory chatMemory, ChatClient.Builder builder) {
        this.chatMemory = chatMemory;
        this.chatClient = builder.build();
    }

    /**
     * @param userId 用作会话 ID，同一个 userId 共享上下文
     */
    @GetMapping("/chat")
    public ResponseEntity<Flux<String>> fluxChat(@RequestParam String message,
                                                 @RequestParam String userId) {
        MessageChatMemoryAdvisor memoryAdvisor =
                MessageChatMemoryAdvisor.builder(chatMemory).build();

        Flux<String> body = chatClient.prompt()
                .system("""
                        你是一个热情、耐心的中文 AI 助手。
                        回答时语气友好自然，可以先简短回应对方的问题，再给出清晰、具体的答案。
                        表达简洁有条理，必要时才分点，不要堆砌术语。
                        不确定的地方坦率说明，不要编造。
                        """)
                .user(message)
                .advisors(memoryAdvisor)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, userId))
                .stream()
                .content();

        return ResponseEntity.ok()
                .contentType(UTF8_EVENT_STREAM)
                .body(body);
    }
}
