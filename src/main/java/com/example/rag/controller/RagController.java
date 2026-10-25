package com.example.rag.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.rag.entity.ParentBlock;
import com.example.rag.mapper.ParentBlockMapper;
import com.example.rag.service.VectorIndexService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 知识库写入与检索问答。
 *
 * <p>检索采用「父子分块 + 多路召回」：</p>
 * <ol>
 *   <li>子块（500 token）入 Redis 向量库，负责语义召回；</li>
 *   <li>父块（2000 token）存 MySQL，负责关键词召回，同时作为最终喂给模型的上下文；</li>
 *   <li>两路结果用 RRF 融合，取分数最高的若干个父块拼上下文。</li>
 * </ol>
 */
@RestController
public class RagController {

    private static final Logger log = LoggerFactory.getLogger(RagController.class);

    private static final DateTimeFormatter TIMESTAMP_FORMATTER =
            DateTimeFormatter.ofPattern("yyyyMMddHHmmssSSS");

    /** 向量召回条数（子块粒度） */
    private static final int VECTOR_TOP_K = 8;
    /** 关键词召回条数（父块粒度） */
    private static final int KEYWORD_TOP_K = 8;
    /** 融合后最终使用的父块数量 */
    private static final int FUSED_TOP_N = 4;
    /** RRF 平滑系数，经典取值 60 */
    private static final int RRF_K = 60;
    /** 拼进 Prompt 的最大字符数，避免超出模型上下文 */
    private static final int MAX_CONTEXT_CHARS = 6000;

    private static final int PARENT_CHUNK_SIZE = 2000;
    private static final int PARENT_MIN_CHUNK_CHARS = 1500;
    private static final int CHILD_CHUNK_SIZE = 500;
    private static final int CHILD_MIN_CHUNK_CHARS = 300;

    private final VectorStore vectorStore;
    private final ChatClient chatClient;
    private final ParentBlockMapper parentBlockMapper;
    private final VectorIndexService vectorIndexService;

    public RagController(VectorStore vectorStore,
                         ChatClient.Builder builder,
                         ParentBlockMapper parentBlockMapper,
                         VectorIndexService vectorIndexService) {
        this.vectorStore = vectorStore;
        this.chatClient = builder.build();
        this.parentBlockMapper = parentBlockMapper;
        this.vectorIndexService = vectorIndexService;
    }

    // ==================== 写入 ====================

    /**
     * 手动录入一段文本到知识库。
     */
    @PostMapping("/rag/upload")
    public String upload(@RequestParam String context,
                         @RequestParam(defaultValue = "u1") String userId,
                         @RequestParam String kbId) {
        if (context == null || context.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "内容不能为空");
        }

        String parentId = buildParentId(userId, kbId, 0);

        ParentBlock block = new ParentBlock();
        block.setParentId(parentId);
        block.setUserId(userId);
        block.setKbId(kbId);
        block.setSource("manual_input");
        block.setContent(context);
        block.setChunkIndex(0);
        block.setCreatedAt(LocalDateTime.now());
        parentBlockMapper.insert(block);

        Document child = new Document(context);
        child.getMetadata().put("parent_id", parentId);
        child.getMetadata().put("source", "manual_input");
        child.getMetadata().put("user_id", userId);
        child.getMetadata().put("kb_id", kbId);
        vectorStore.add(List.of(child));

        return parentId;
    }

    /**
     * 上传 PDF：解析 → 清洗 → 父块入 MySQL → 子块入向量库。
     */
    @PostMapping("/rag/upload-pdf")
    @Transactional(rollbackFor = Exception.class)
    public String uploadPdf(@RequestParam("file") MultipartFile file,
                            @RequestParam String userId,
                            @RequestParam String kbId) throws IOException {
        String fileName = file.getOriginalFilename();
        if (fileName == null || fileName.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "文件名不能为空");
        }

        Path tempFile = Files.createTempFile("rag-", ".pdf");
        try {
            file.transferTo(tempFile);
            Resource resource = new FileSystemResource(tempFile);
            List<Document> pages = new PagePdfDocumentReader(resource).get();

            // PDF 解析出来的文本常带多余空白和换行，先归一化再切分，切片质量更好
            List<Document> cleanedPages = pages.stream()
                    .map(doc -> new Document(doc.getText().replaceAll("\\s+", " ").trim(), doc.getMetadata()))
                    .filter(doc -> !doc.getText().isBlank())
                    .toList();

            TokenTextSplitter parentSplitter = TokenTextSplitter.builder()
                    .withChunkSize(PARENT_CHUNK_SIZE)
                    .withMinChunkSizeChars(PARENT_MIN_CHUNK_CHARS)
                    .build();
            TokenTextSplitter childSplitter = TokenTextSplitter.builder()
                    .withChunkSize(CHILD_CHUNK_SIZE)
                    .withMinChunkSizeChars(CHILD_MIN_CHUNK_CHARS)
                    .build();

            List<Document> parentChunks = parentSplitter.apply(cleanedPages);

            // 同名文件重复上传时，先清掉上一次的父块与向量，避免检索出重复内容
            parentBlockMapper.delete(new LambdaQueryWrapper<ParentBlock>()
                    .eq(ParentBlock::getKbId, kbId)
                    .eq(ParentBlock::getUserId, userId)
                    .eq(ParentBlock::getSource, fileName));
            vectorIndexService.deleteByFilter(
                    "kb_id == '" + safe(kbId) + "' && source == '" + safe(fileName) + "'");

            List<Document> allChildChunks = new ArrayList<>();
            for (int i = 0; i < parentChunks.size(); i++) {
                String parentId = buildParentId(userId, kbId, i);

                ParentBlock block = new ParentBlock();
                block.setSource(fileName);
                block.setParentId(parentId);
                block.setContent(parentChunks.get(i).getText());
                block.setKbId(kbId);
                block.setUserId(userId);
                block.setChunkIndex(i);
                block.setCreatedAt(LocalDateTime.now());
                parentBlockMapper.insert(block);

                for (Document chunk : childSplitter.apply(List.of(parentChunks.get(i)))) {
                    chunk.getMetadata().put("parent_id", parentId);
                    chunk.getMetadata().put("source", fileName);
                    chunk.getMetadata().put("user_id", userId);
                    chunk.getMetadata().put("kb_id", kbId);
                    allChildChunks.add(chunk);
                }
            }
            vectorStore.add(allChildChunks);

            return "上传成功，父块 " + parentChunks.size() + " 个，子块 " + allChildChunks.size() + " 个";
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }

    // ==================== 检索问答 ====================

    /**
     * 多路召回 + RRF 融合 + 生成。
     */
    @PostMapping("/rag/ask")
    public Map<String, Object> ragAsk(@RequestParam String question,
                                      @RequestParam(defaultValue = "u1") String userId,
                                      @RequestParam String kbId) {
        String q = question == null ? "" : question.trim();
        if (q.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "问题不能为空");
        }

        // 路 1：向量召回（子块 → 父块）
        List<String> vectorParents = recallByVector(q, userId, kbId);

        // 路 2：关键词召回（父块全文索引）
        List<String> keywordParents = recallByKeyword(q, userId, kbId);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("question", q);
        result.put("recall", Map.of(
                "vector", vectorParents.size(),
                "keyword", keywordParents.size()));

        if (vectorParents.isEmpty() && keywordParents.isEmpty()) {
            result.put("answer", "抱歉，知识库里没有找到与这个问题相关的内容。");
            result.put("sources", List.of());
            return result;
        }

        List<String> fusedParentIds = rrfFuse(vectorParents, keywordParents);

        // 按融合后的顺序取父块内容
        Map<String, ParentBlock> blockMap = parentBlockMapper.selectList(
                        new LambdaQueryWrapper<ParentBlock>()
                                .in(ParentBlock::getParentId, fusedParentIds)
                                .eq(ParentBlock::getKbId, kbId)
                                .eq(ParentBlock::getUserId, userId))
                .stream()
                .collect(Collectors.toMap(ParentBlock::getParentId, b -> b, (a, b) -> b));

        StringBuilder context = new StringBuilder();
        List<Map<String, Object>> sources = new ArrayList<>();
        for (String parentId : fusedParentIds) {
            ParentBlock block = blockMap.get(parentId);
            if (block == null || block.getContent() == null) {
                continue;
            }
            if (context.length() + block.getContent().length() > MAX_CONTEXT_CHARS) {
                break;
            }
            context.append(block.getContent()).append("\n\n---\n\n");

            Map<String, Object> source = new LinkedHashMap<>();
            source.put("parentId", parentId);
            source.put("file", block.getSource());
            source.put("preview", preview(block.getContent()));
            sources.add(source);
        }

        if (context.isEmpty()) {
            result.put("answer", "抱歉，知识库里没有找到与这个问题相关的内容。");
            result.put("sources", List.of());
            return result;
        }

        String answer = chatClient.prompt()
                .system("""
                        你是一个知识库助手，规则如下：
                        1. 只能使用【资料】部分的内容回答问题。
                        2. 如果【资料】中没有相关信息，必须直接回答"资料中没有找到相关信息"。
                        3. 禁止使用你自己的知识。
                        4. 回答用中文，条理清晰；涉及原文数据时尽量引用原句。
                        """)
                .user("【资料】\n" + context + "\n问题：\n" + q)
                .call()
                .content();

        result.put("answer", answer);
        result.put("sources", sources);
        return result;
    }

    /**
     * 向量召回：子块命中后按 parent_id 去重，保持相似度排序。
     */
    private List<String> recallByVector(String question, String userId, String kbId) {
        try {
            List<Document> hits = vectorStore.similaritySearch(SearchRequest.builder()
                    .query(question)
                    .topK(VECTOR_TOP_K)
                    .similarityThreshold(0.6)
                    .filterExpression("user_id == '" + safe(userId) + "' && kb_id == '" + safe(kbId) + "'")
                    .build());

            if (hits == null) {
                return List.of();
            }
            LinkedHashSet<String> parents = new LinkedHashSet<>();
            for (Document hit : hits) {
                Object parentId = hit.getMetadata().get("parent_id");
                if (parentId != null) {
                    parents.add(parentId.toString());
                }
            }
            return new ArrayList<>(parents);
        } catch (Exception e) {
            log.warn("向量召回失败，本轮只使用关键词召回", e);
            return List.of();
        }
    }

    /**
     * 关键词召回：命中 MySQL 全文索引。
     *
     * <p>先用「去掉疑问词后的核心词」做精确匹配（ngram 下效果最好），
     * 没命中再退回整句自然语言检索。索引不存在或分词不可用时，
     * 这一路整体降级为空，不影响向量召回。</p>
     */
    private List<String> recallByKeyword(String question, String userId, String kbId) {
        try {
            String keyword = toKeyword(question);
            if (!keyword.isBlank()) {
                List<String> exact = parentBlockMapper.searchByKeyword(userId, kbId, keyword, KEYWORD_TOP_K);
                if (exact != null && !exact.isEmpty()) {
                    return exact;
                }
            }
            List<String> fuzzy = parentBlockMapper.searchByQuestionNatural(userId, kbId, question, KEYWORD_TOP_K);
            return fuzzy == null ? List.of() : fuzzy;
        } catch (Exception e) {
            log.warn("关键词召回失败（可能是全文索引未创建），本轮降级为纯向量召回", e);
            return List.of();
        }
    }

    /**
     * Reciprocal Rank Fusion：score = Σ 1 / (k + rank)，rank 从 1 开始。
     * 两路的分数不可直接比较，但名次可以，所以用名次而不是原始分数来融合。
     */
    private List<String> rrfFuse(List<String> vectorParents, List<String> keywordParents) {
        Map<String, Double> score = new HashMap<>();
        for (int i = 0; i < vectorParents.size(); i++) {
            score.merge(vectorParents.get(i), 1.0 / (RRF_K + i + 1), Double::sum);
        }
        for (int i = 0; i < keywordParents.size(); i++) {
            score.merge(keywordParents.get(i), 1.0 / (RRF_K + i + 1), Double::sum);
        }
        return score.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(FUSED_TOP_N)
                .map(Map.Entry::getKey)
                .toList();
    }

    /**
     * 问句里不影响检索的口语成分。按长度从长到短排列，避免先删短词把长词拆坏。
     */
    private static final List<String> QUESTION_NOISE = List.of(
            "的内容是什么", "的内容是", "内容是什么", "的内容", "是什么样", "是什么", "是多少",
            "什么意思", "讲什么", "说了什么", "怎么规定", "如何规定", "规定是什么", "有哪些内容",
            "有哪些", "请问", "我想知道", "告诉我", "能不能", "可不可以", "什么时候", "怎么",
            "如何", "为什么", "哪些", "吗", "呢", "呀", "啊");

    /**
     * 把问句压成检索词。
     *
     * <p>例：「第一百零九条的内容是什么」→「第一百零九条」。
     * 核心词越短，ngram 的 2-gram 组合越可能全部命中，精确度越高。</p>
     */
    private String toKeyword(String question) {
        String s = question;
        for (String noise : QUESTION_NOISE) {
            s = s.replace(noise, "");
        }
        // 只保留汉字、字母、数字，其余（标点、布尔运算符）一律去掉
        s = s.replaceAll("[^\\p{IsHan}A-Za-z0-9]", "").trim();
        return s.length() > 24 ? s.substring(0, 24) : s;
    }

    private String preview(String content) {
        String flat = content.replaceAll("\\s+", " ").trim();
        return flat.length() <= 120 ? flat : flat.substring(0, 120) + "…";
    }

    private String buildParentId(String userId, String kbId, int index) {
        return userId + "_" + kbId + "_" + LocalDateTime.now().format(TIMESTAMP_FORMATTER) + "_" + index;
    }

    /** 过滤表达式里出现单引号会把语法打断，这里做最小化处理 */
    private String safe(String value) {
        return value == null ? "" : value.replace("'", "");
    }
}
