package com.example.rag.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.rag.entity.ParentBlock;
import com.example.rag.mapper.ParentBlockMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 查询改写：把口语化的问题翻译成"知识库里可能出现的正式表述"。
 *
 * <p>这里刻意不写任何领域提示词（不写"你是律师"）。领域信息是**从数据里采样**出来的：
 * 取当前知识库的几段正文塞进提示词，模型自己就能判断这批文档的术语体系。
 * 同一套代码换到员工手册、产品文档、合同库都能用。</p>
 */
@Service
public class QueryRewriteService {

    private static final Logger log = LoggerFactory.getLogger(QueryRewriteService.class);

    /** 采样几段正文作为"领域样例" */
    private static final int SAMPLE_COUNT = 3;
    /** 每段截取多少字 */
    private static final int SAMPLE_CHARS = 150;
    /** 最多采用几条改写 */
    private static final int MAX_QUERIES = 3;

    private final ChatClient chatClient;
    private final ParentBlockMapper parentBlockMapper;

    public QueryRewriteService(ChatClient.Builder builder, ParentBlockMapper parentBlockMapper) {
        this.chatClient = builder.build();
        this.parentBlockMapper = parentBlockMapper;
    }

    /**
     * @return 改写后的检索词，失败或无法改写时返回空列表（调用方退回原问题）
     */
    public List<String> rewrite(String question, String userId, String kbId) {
        try {
            String samples = sampleContent(userId, kbId);
            if (samples.isBlank()) {
                return List.of();
            }

            String raw = chatClient.prompt()
                    .system("""
                            你是检索助手。你的任务是把用户的口语化问题，改写成适合在知识库中检索的查询词。

                            规则：
                            1. 参考下面给出的知识库片段，判断这批文档的语言风格和术语体系；
                            2. 输出 3 行，每行一条改写，不要编号，不要解释，不要引号；
                            3. 每条 3~12 个字，尽量使用文档中可能出现的正式表述；
                            4. 如果原问题已经足够具体，就原样输出它；
                            5. 只输出这 3 行，不要输出任何其它内容。

                            知识库片段：
                            %s
                            """.formatted(samples))
                    .user(question)
                    .call()
                    .content();

            return parse(raw);
        } catch (Exception e) {
            log.warn("查询改写失败，本轮退回原问题检索", e);
            return List.of();
        }
    }

    /**
     * 从当前知识库随机取几段正文。取不到说明库是空的，直接放弃改写。
     */
    private String sampleContent(String userId, String kbId) {
        List<ParentBlock> samples = parentBlockMapper.selectList(new LambdaQueryWrapper<ParentBlock>()
                .eq(ParentBlock::getUserId, userId)
                .eq(ParentBlock::getKbId, kbId)
                .last("ORDER BY RAND() LIMIT " + SAMPLE_COUNT));

        StringBuilder sb = new StringBuilder();
        for (ParentBlock sample : samples) {
            if (sample.getContent() == null) {
                continue;
            }
            String text = sample.getContent().replaceAll("\\s+", " ").trim();
            if (text.isEmpty()) {
                continue;
            }
            sb.append("- ")
                    .append(text.length() > SAMPLE_CHARS ? text.substring(0, SAMPLE_CHARS) : text)
                    .append("\n");
        }
        return sb.toString().trim();
    }

    /**
     * 小模型输出 JSON 容易坏，这里用"一行一条"的格式，解析成本最低也最稳。
     */
    private List<String> parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return List.of();
        }
        Set<String> result = new LinkedHashSet<>();
        for (String line : raw.split("\\R")) {
            String cleaned = line.replaceAll("^[\\s\\-•·*\\d.、)）\"'“”]+", "")
                    .replaceAll("[\\s\"'“”，。；;]+$", "")
                    .trim();
            if (cleaned.isEmpty() || cleaned.length() > 24) {
                continue;
            }
            // 模型有时候会把提示语也抄进来
            if (cleaned.contains("改写") || cleaned.contains("规则") || cleaned.contains("知识库")) {
                continue;
            }
            result.add(cleaned);
            if (result.size() >= MAX_QUERIES) {
                break;
            }
        }
        return new ArrayList<>(result);
    }
}
