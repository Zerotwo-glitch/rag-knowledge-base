package com.example.rag.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.rag.entity.ParentBlock;
import com.example.rag.mapper.ParentBlockMapper;
import com.example.rag.service.QueryRewriteService;
import com.example.rag.service.VectorIndexService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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
    /**
     * 关键词路的权重。
     *
     * <p>关键词是"精确命中"，比向量的"语义接近"更可信，所以给它加权；
     * 不加权时它和向量第一名同为 1/(60+1)，同分排序取决于哈希顺序，
     * 结果就是精确命中的条文时灵时不灵。</p>
     */
    private static final double KEYWORD_WEIGHT = 1.5;
    /**
     * 超过这个长度就不再做精确关键词匹配。
     *
     * <p>ngram 的 BOOLEAN 模式要求检索词的所有 2-gram 都命中，一长串口语
     * （「我在网上买了个手机快递显示已经签收」）在原文档里几乎不可能原样出现，
     * 命中率接近 0，白白多打一次数据库。这类问题交给查询改写去覆盖。</p>
     */
    private static final int MAX_KEYWORD_LENGTH = 12;
    /** 拼进 Prompt 的最大字符数，避免超出模型上下文 */
    private static final int MAX_CONTEXT_CHARS = 8000;

    private static final int PARENT_CHUNK_SIZE = 2000;
    private static final int PARENT_MIN_CHUNK_CHARS = 1500;
    private static final int CHILD_CHUNK_SIZE = 500;
    private static final int CHILD_MIN_CHUNK_CHARS = 300;

    /** 统一的不命中话术：说明边界，并把文档外的问题引导到对话页 */
    private static final String NOT_FOUND =
            "资料中没有找到相关信息。知识库只回答文档里的内容，其他问题可以到「多轮对话」页提问。";

    private final VectorStore vectorStore;
    private final ChatClient chatClient;
    private final ParentBlockMapper parentBlockMapper;
    private final VectorIndexService vectorIndexService;
    private final QueryRewriteService queryRewriteService;

    /** 向量召回的最低相似度 */
    private final double vectorThreshold;
    /** 认为"资料能回答这个问题"的最低相似度 */
    private final double confidentScore;
    /** 是否允许低置信时触发查询改写 */
    private final boolean rewriteEnabled;

    public RagController(VectorStore vectorStore,
                         ChatClient.Builder builder,
                         ParentBlockMapper parentBlockMapper,
                         VectorIndexService vectorIndexService,
                         QueryRewriteService queryRewriteService,
                         @Value("${app.rag.vector-threshold:0.6}") double vectorThreshold,
                         @Value("${app.rag.confident-score:0.75}") double confidentScore,
                         @Value("${app.rag.rewrite-enabled:true}") boolean rewriteEnabled) {
        this.vectorStore = vectorStore;
        this.chatClient = builder.build();
        this.parentBlockMapper = parentBlockMapper;
        this.vectorIndexService = vectorIndexService;
        this.queryRewriteService = queryRewriteService;
        this.vectorThreshold = vectorThreshold;
        this.confidentScore = confidentScore;
        this.rewriteEnabled = rewriteEnabled;
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

            // PDF 是按页返回 Document 的，切分器不会跨文档合并。
            // 直接切分的话每个父块只有一页（几百字），父子分块就退化成了"页块"。
            // 所以先把全文拼成一份再切，父块才真的是 2000 token 级别。
            String fullText = cleanedPages.stream()
                    .map(Document::getText)
                    .collect(Collectors.joining("\n"));
            Document merged = new Document(fullText);
            merged.getMetadata().put("source", fileName);

            List<Document> parentChunks = parentSplitter.apply(List.of(merged));

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

        // 寒暄类输入没有信息需求，不该走检索：既浪费一次召回，
        // 又会把无关段落塞进上下文，模型还容易顺着"资料"说出莫名其妙的话。
        if (isSmallTalk(q)) {
            Map<String, Object> greeting = new LinkedHashMap<>();
            greeting.put("question", q);
            greeting.put("rewritten", List.of());
            greeting.put("recall", Map.of("vector", 0, "keyword", 0, "topScore", 0.0));
            greeting.put("answer", "您好，我是这个知识库的问答助手。直接问文档里的内容就行，比如「第一百零九条讲了什么」。");
            greeting.put("sources", List.of());
            return greeting;
        }

        // 第一轮：拿原问题直接召回
        Recall recall = recall(List.of(q), userId, kbId);
        List<String> rewritten = List.of();

        // 置信度不够，才用查询改写再试一轮（多花一次模型调用，只在必要时触发）
        if (!isConfident(recall) && rewriteEnabled) {
            rewritten = queryRewriteService.rewrite(q, userId, kbId);
            if (!rewritten.isEmpty()) {
                List<String> queries = new ArrayList<>();
                queries.add(q);
                queries.addAll(rewritten);
                recall = recall(queries, userId, kbId);
            }
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("question", q);
        result.put("rewritten", rewritten);
        result.put("recall", Map.of(
                "vector", recall.vectorCount(),
                "keyword", recall.keywordCount(),
                "topScore", Math.round(recall.topScore() * 100) / 100.0));

        // 两轮都不够可信：直接拒答，不调生成模型。
        // 否则模型会拿不相关的资料硬凑答案，这比"没查到"危险得多。
        if (!isConfident(recall)) {
            result.put("answer", NOT_FOUND);
            result.put("sources", List.of());
            return result;
        }

        List<String> fusedParentIds = recall.fusedIds();

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
        Set<String> usedParentIds = new HashSet<>(fusedParentIds);

        for (String parentId : fusedParentIds) {
            ParentBlock block = blockMap.get(parentId);
            if (block == null || block.getContent() == null) {
                continue;
            }

            // 条文可能正好被切在两个父块之间：命中块里只有开头，续接块里才是后半段。
            // 所以命中后把「同来源的下一个父块」一并带上，避免答案只看到半条。
            String text = block.getContent();
            if (looksTruncated(text)) {
                ParentBlock next = findNextBlock(block, userId, kbId);
                if (next != null && next.getContent() != null && !usedParentIds.contains(next.getParentId())) {
                    text = text + "\n" + next.getContent();
                    usedParentIds.add(next.getParentId());
                }
            }

            if (context.length() + text.length() > MAX_CONTEXT_CHARS && !sources.isEmpty()) {
                break;
            }
            context.append(text).append("\n\n---\n\n");

            Map<String, Object> source = new LinkedHashMap<>();
            source.put("parentId", parentId);
            source.put("file", block.getSource());
            source.put("preview", preview(text));
            sources.add(source);
        }

        if (context.isEmpty()) {
            result.put("answer", NOT_FOUND);
            result.put("sources", List.of());
            return result;
        }

        String answer = chatClient.prompt()
                .system("""
                        你是这个知识库的问答助手。
                        只依据【资料】里的内容回答；资料里没有的，直接说"资料中没有找到相关信息"，
                        不要用你自己的知识补充，也不要推测。
                        回答用中文，简洁直接，先给结论；涉及原文数据时尽量引用原句。
                        不要在回答里提到这些说明。
                        """)
                .user("【资料】\n" + context + "\n问题：\n" + q)
                .call()
                .content();

        result.put("answer", answer);
        result.put("sources", sources);
        return result;
    }

    /**
     * 一轮召回的结果。
     *
     * @param fusedIds     跨列表 RRF 融合后的父块 id（已按分数排序）
     * @param vectorCount  向量路命中的父块数
     * @param keywordCount 关键词路命中的父块数
     * @param topScore     向量路最高相似度（只统计真实存在的父块）
     */
    private record Recall(List<String> fusedIds, int vectorCount, int keywordCount, double topScore) {
    }

    /**
     * 对一批查询做召回，并用 RRF 融合。
     *
     * <p>关键点：**每个查询的每条召回路径各自是一个有序列表**，分别参与 RRF，
     * 而不是把所有结果按查询顺序首尾相接。后者会毁掉名次——原问题的第 8 名
     * 会排到改写查询的第 1 名之前，而 RRF 只看名次不看原始分数。</p>
     */
    private Recall recall(List<String> queries, String userId, String kbId) {
        List<List<String>> vectorLists = new ArrayList<>();
        List<List<String>> keywordLists = new ArrayList<>();
        Map<String, Double> scoreByParent = new HashMap<>();
        Set<String> vectorParents = new LinkedHashSet<>();
        Set<String> keywordParents = new LinkedHashSet<>();

        for (String query : queries) {
            // 向量路：同一查询内按 parent 去重，保持相似度顺序
            List<String> vectorList = new ArrayList<>();
            for (Document hit : vectorSearch(query, userId, kbId)) {
                Object parentId = hit.getMetadata().get("parent_id");
                if (parentId == null) {
                    continue;
                }
                String pid = parentId.toString();
                if (vectorList.contains(pid)) {
                    continue;
                }
                vectorList.add(pid);
                if (hit.getScore() != null) {
                    scoreByParent.merge(pid, hit.getScore(), Math::max);
                }
            }
            if (!vectorList.isEmpty()) {
                vectorLists.add(vectorList);
                vectorParents.addAll(vectorList);
            }

            List<String> keywordList = keywordSearch(query, userId, kbId);
            if (!keywordList.isEmpty()) {
                keywordLists.add(keywordList);
                keywordParents.addAll(keywordList);
            }
        }

        // 向量库里可能残留"父块已被删除/被重传替换"的孤儿向量。
        // 不过滤的话，它们会占满 top-N，表现为"明明召回了 20 多条，却一个父块都取不到"。
        Set<String> alive = existingParentIds(vectorParents, keywordParents, userId, kbId);
        vectorParents.retainAll(alive);
        keywordParents.retainAll(alive);

        double topScore = scoreByParent.entrySet().stream()
                .filter(e -> alive.contains(e.getKey()))
                .mapToDouble(Map.Entry::getValue)
                .max()
                .orElse(0.0);

        // 跨列表 RRF：同一父块被多个查询命中时，分数自然累加。
        // 关键词路加权——精确命中应当压过"语义接近"的向量命中。
        Map<String, Double> fused = new HashMap<>();
        for (List<String> list : vectorLists) {
            accumulate(fused, list, alive, 1.0);
        }
        for (List<String> list : keywordLists) {
            accumulate(fused, list, alive, KEYWORD_WEIGHT);
        }
        List<String> fusedIds = fused.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(FUSED_TOP_N)
                .map(Map.Entry::getKey)
                .toList();

        return new Recall(fusedIds, vectorParents.size(), keywordParents.size(), topScore);
    }

    /**
     * 把一个有序列表按 RRF 累加进融合分数（可加权）。
     */
    private void accumulate(Map<String, Double> fused, List<String> list, Set<String> alive, double weight) {
        int rank = 0;
        for (String pid : list) {
            if (!alive.contains(pid)) {
                continue;
            }
            rank++;
            fused.merge(pid, weight / (RRF_K + rank), Double::sum);
        }
    }

    /**
     * 查出这批 id 里真正还在 MySQL 的父块。
     */
    private Set<String> existingParentIds(Collection<String> a, Collection<String> b, String userId, String kbId) {
        Set<String> candidates = new LinkedHashSet<>(a);
        candidates.addAll(b);
        if (candidates.isEmpty()) {
            return Set.of();
        }
        return parentBlockMapper.selectList(new LambdaQueryWrapper<ParentBlock>()
                        .select(ParentBlock::getParentId)
                        .in(ParentBlock::getParentId, candidates)
                        .eq(ParentBlock::getKbId, kbId)
                        .eq(ParentBlock::getUserId, userId))
                .stream()
                .map(ParentBlock::getParentId)
                .collect(Collectors.toSet());
    }

    /**
     * 置信度判断。
     *
     * <p>不能只看"有没有返回结果"——向量检索几乎永远会返回 topK 条，
     * 哪怕全是无关内容。所以要么精确关键词命中，要么向量最高相似度过线，
     * 才算"资料能回答这个问题"。</p>
     */
    private boolean isConfident(Recall recall) {
        return recall.keywordCount() > 0 || recall.topScore() >= confidentScore;
    }

    /**
     * 向量召回：子块命中后返回原始 Document（需要读取 similarity 分数）。
     */
    private List<Document> vectorSearch(String query, String userId, String kbId) {
        try {
            List<Document> hits = vectorStore.similaritySearch(SearchRequest.builder()
                    .query(query)
                    .topK(VECTOR_TOP_K)
                    .similarityThreshold(vectorThreshold)
                    .filterExpression("user_id == '" + safe(userId) + "' && kb_id == '" + safe(kbId) + "'")
                    .build());
            return hits == null ? List.of() : hits;
        } catch (Exception e) {
            log.warn("向量召回失败，本轮只使用关键词召回：{}", e.getMessage());
            return List.of();
        }
    }

    /**
     * 关键词召回：只保留「核心词精确命中」这一条路。
     *
     * <p>早先加过整句自然语言模式的兜底，但那个模式无论相关性多低都会返回 topK 条，
     * 而 RRF 只看名次不看分数，噪声会直接把融合结果带偏。所以要召回精度，
     * 要覆盖率就交给上面的查询改写。</p>
     */
    private List<String> keywordSearch(String query, String userId, String kbId) {
        String keyword = toKeyword(query);
        if (keyword.isBlank() || keyword.length() > MAX_KEYWORD_LENGTH) {
            return List.of();
        }
        try {
            List<String> parents = parentBlockMapper.searchByKeyword(userId, kbId, keyword, KEYWORD_TOP_K);
            return parents == null ? List.of() : parents;
        } catch (Exception e) {
            log.warn("关键词召回失败（可能是全文索引未创建），本轮降级为纯向量召回：{}", e.getMessage());
            return List.of();
        }
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
        // 条文引用是最常见的一类问法，直接抽出来，别让「内容」「是什么」这些尾巴参与匹配。
        // 多一个「内容」就会多出「条内」这个 2-gram，而原文里没有，整句直接 0 命中。
        String compact = question.replaceAll("[^\\p{IsHan}A-Za-z0-9]", "");

        Matcher arabic = ARTICLE_REF.matcher(compact);
        if (arabic.find()) {
            return "第" + toChineseNumber(Integer.parseInt(arabic.group(1))) + "条";
        }
        Matcher chinese = ARTICLE_REF_CN.matcher(compact);
        if (chinese.find()) {
            return "第" + chinese.group(1) + "条";
        }

        String s = question;
        for (String noise : QUESTION_NOISE) {
            s = s.replace(noise, "");
        }
        // 只保留汉字、字母、数字，其余（标点、布尔运算符）一律去掉
        s = s.replaceAll("[^\\p{IsHan}A-Za-z0-9]", "").trim();
        return s.length() > 24 ? s.substring(0, 24) : s;
    }

    /**
     * 匹配「第123条」这样的阿拉伯数字条文引用。
     */
    private static final Pattern ARTICLE_REF = Pattern.compile("第\\s*(\\d{1,4})\\s*条");

    /**
     * 匹配「第一百零九条」这样的中文数字条文引用。
     */
    private static final Pattern ARTICLE_REF_CN = Pattern.compile("第([零一二三四五六七八九十百千两]{1,8})条");

    private static final String[] DIGITS = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};

    /**
     * 阿拉伯数字转中文数字（1~9999），只为了拼条文号。
     */
    private String toChineseNumber(int n) {
        if (n <= 0) {
            return "零";
        }
        int[] units = {1000, 100, 10, 1};
        String[] names = {"千", "百", "十", ""};
        StringBuilder sb = new StringBuilder();
        boolean started = false;
        boolean zeroPending = false;

        for (int i = 0; i < units.length; i++) {
            int digit = n / units[i] % 10;
            if (digit == 0) {
                if (started) {
                    zeroPending = true;
                }
                continue;
            }
            if (zeroPending) {
                sb.append('零');
                zeroPending = false;
            }
            sb.append(DIGITS[digit]).append(names[i]);
            started = true;
        }
        String result = sb.toString();
        // 10~19 写成「十X」而不是「一十X」
        return result.startsWith("一十") ? result.substring(1) : result;
    }

    /**
     * 块尾不是句末标点，基本可以判定这段被切断了（页码「－104－」同理）。
     * 只有判断为被切断时才去接下一个父块，避免无脑扩容把上下文预算吃光。
     */
    private boolean looksTruncated(String text) {
        String t = text == null ? "" : text.strip();
        if (t.isEmpty()) {
            return false;
        }
        return !t.matches("(?s).*[。！？；!?;]$");
    }

    /**
     * 寒暄词表：只做整句匹配，避免「合同」这种两字实词被误伤。
     */
    private static final Set<String> SMALL_TALK = Set.of(
            "你好", "您好", "hi", "hello", "在吗", "在么", "在不在", "有人吗",
            "谢谢", "多谢", "感谢", "再见", "拜拜", "你是谁", "你叫什么",
            "测试", "test", "11", "1");

    /**
     * 判断是不是寒暄/占位输入。
     */
    private boolean isSmallTalk(String question) {
        String normalized = question.replaceAll("[\\s\\p{P}\\p{S}]", "").toLowerCase();
        return normalized.isEmpty() || SMALL_TALK.contains(normalized);
    }

    /**
     * 取同来源、序号 +1 的父块，也就是原文里紧接在后面的那一段。
     */
    private ParentBlock findNextBlock(ParentBlock block, String userId, String kbId) {
        if (block.getChunkIndex() == null) {
            return null;
        }
        return parentBlockMapper.selectOne(new LambdaQueryWrapper<ParentBlock>()
                .eq(ParentBlock::getUserId, userId)
                .eq(ParentBlock::getKbId, kbId)
                .eq(ParentBlock::getSource, block.getSource())
                .eq(ParentBlock::getChunkIndex, block.getChunkIndex() + 1)
                .last("limit 1"));
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
