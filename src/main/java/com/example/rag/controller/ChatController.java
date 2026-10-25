package com.example.rag.controller;

import com.example.rag.entity.User;
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

@RestController
public class ChatController {
    private static final MediaType UTF8_EVENT_STREAM =
            new MediaType("text", "event-stream", StandardCharsets.UTF_8);
    private final ChatMemory chatMemory;
    private final ChatClient chatClient;

    public ChatController(ChatMemory chatMemory,ChatClient.Builder builder) {
        this.chatMemory = chatMemory;
        this.chatClient = builder.build();
    }

    @GetMapping("/chat")
    public ResponseEntity<Flux<String>> fluxChat(@RequestParam String message,
                                                 @RequestParam String userId) {
        MessageChatMemoryAdvisor memoryAdvisor =
                MessageChatMemoryAdvisor.builder(chatMemory).build();

        Flux<String> body = chatClient.prompt()
                .system("你是一个严厉的Java面试官，围绕Spring Boot、MySQL、Redis对我进行连环追问，每次只问一个问题，等我回答后再问下一个。")
                .user(message)
                .advisors(memoryAdvisor)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, userId))
                .stream()
                .content();

        return ResponseEntity.ok()
                .contentType(UTF8_EVENT_STREAM)
                .body(body);
    }
    @GetMapping("/extract")
    public Object extractChat(@RequestParam String message) {
        User user = chatClient.prompt()
                .system("""
                        你是一个信息提取助手，只能根据用户提问提取信息，不可随意编造数据。
                        如果提取不到 age 填：-1
                        如果提取不到 name 填：未知
                        """)
                .user(message)
                .call()
                .entity(User.class);

        boolean ageMissing = user.getAge() == null || user.getAge() == -1;
        boolean nameMissing = user.getName() == null || "未知".equals(user.getName());

        if (ageMissing && nameMissing) return "未提取到年龄和姓名信息";
        if (ageMissing) return "未提取到年龄信息";
        if (nameMissing) return "未提取到姓名信息";
        return user;
    }
}
