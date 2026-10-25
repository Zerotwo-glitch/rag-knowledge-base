package com.example.rag.config;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.redis.RedisVectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import redis.clients.jedis.RedisClient;

@Configuration
public class VectorStoreConfig {

    @Value("${spring.data.redis.host}")
    private String redisHost;

    @Value("${spring.data.redis.port}")
    private int redisPort;

    @Bean
    public RedisClient redisClient() {
        return RedisClient.builder()
                .hostAndPort(redisHost, redisPort)
                .build();
    }

    @Bean
    public VectorStore vectorStore(RedisClient redisClient, EmbeddingModel embeddingModel) {
        return RedisVectorStore.builder(redisClient, embeddingModel)
                .indexName("rag-index")
                .prefix("rag:")
                .metadataFields(
                        RedisVectorStore.MetadataField.tag("parent_id"),
                        RedisVectorStore.MetadataField.tag("source"),
                        RedisVectorStore.MetadataField.tag("file_name"),
                        RedisVectorStore.MetadataField.tag("user_id"),
                        RedisVectorStore.MetadataField.tag("kb_id"),
                        RedisVectorStore.MetadataField.numeric("chunk_index"),
                        RedisVectorStore.MetadataField.numeric("page_number")
                )
                .initializeSchema(true)
                .build();
    }
}