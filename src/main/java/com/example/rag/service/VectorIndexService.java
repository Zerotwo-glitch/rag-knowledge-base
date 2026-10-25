package com.example.rag.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 向量库维护工具。
 *
 * <p>Spring AI 的 {@link VectorStore} 只提供"按 id 删除"，没有"按条件批量删除"。
 * 直接写 topK(10000) 会带来两个问题：一是一次拉全量可能被截断，二是数据量大时内存吃紧。
 * 这里统一用"分批检索 → 按 id 删除"的方式，并在方法内限制最大轮数。</p>
 */
@Service
public class VectorIndexService {

    private static final Logger log = LoggerFactory.getLogger(VectorIndexService.class);

    /** 每批拉取数量 */
    private static final int BATCH_SIZE = 500;

    /** 最多循环轮数，防止过滤条件写错时死循环 */
    private static final int MAX_ROUNDS = 20;

    private final VectorStore vectorStore;

    public VectorIndexService(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /**
     * 删除满足过滤条件的全部向量。
     *
     * @param filterExpression 例如 "kb_id == 'xxx'" 或 "source == 'a.pdf'"
     * @return 实际删除的向量数量
     */
    public int deleteByFilter(String filterExpression) {
        int deleted = 0;
        for (int round = 0; round < MAX_ROUNDS; round++) {
            List<Document> docs = vectorStore.similaritySearch(SearchRequest.builder()
                    .query("")
                    .topK(BATCH_SIZE)
                    .similarityThreshold(0.0)
                    .filterExpression(filterExpression)
                    .build());

            if (docs == null || docs.isEmpty()) {
                break;
            }
            List<String> ids = docs.stream().map(Document::getId).toList();
            vectorStore.delete(ids);
            deleted += ids.size();

            // 这一批没拉满，说明已经删干净了
            if (ids.size() < BATCH_SIZE) {
                break;
            }
        }
        if (deleted > 0) {
            log.info("按条件删除向量 {} 条，条件：{}", deleted, filterExpression);
        }
        return deleted;
    }
}
