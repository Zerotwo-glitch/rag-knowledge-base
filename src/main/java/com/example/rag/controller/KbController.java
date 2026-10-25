package com.example.rag.controller;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.example.rag.entity.KnowledgeBase;
import com.example.rag.entity.ParentBlock;
import com.example.rag.mapper.KnowledgeBaseMapper;
import com.example.rag.mapper.ParentBlockMapper;
import com.example.rag.service.VectorIndexService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * 知识库的增删查。
 */
@RestController
public class KbController {

    private static final Logger log = LoggerFactory.getLogger(KbController.class);

    private final ParentBlockMapper parentBlockMapper;
    private final KnowledgeBaseMapper knowledgeBaseMapper;
    private final VectorIndexService vectorIndexService;

    public KbController(ParentBlockMapper parentBlockMapper,
                        KnowledgeBaseMapper knowledgeBaseMapper,
                        VectorIndexService vectorIndexService) {
        this.parentBlockMapper = parentBlockMapper;
        this.knowledgeBaseMapper = knowledgeBaseMapper;
        this.vectorIndexService = vectorIndexService;
    }

    @PostMapping("/kb/create")
    public String createKnowledgeBase(@RequestParam String userId,
                                      @RequestParam String name) {
        if (name == null || name.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "知识库名称不能为空");
        }
        String kbId = UUID.randomUUID().toString();

        KnowledgeBase kb = new KnowledgeBase();
        kb.setId(kbId);
        kb.setUserId(userId);
        kb.setName(name);
        kb.setCreatedAt(LocalDateTime.now());
        knowledgeBaseMapper.insert(kb);

        return kbId;
    }

    /**
     * 查询用户的所有知识库
     */
    @GetMapping("/kb/list")
    public List<KnowledgeBase> listKnowledgeBases(@RequestParam String userId) {
        return knowledgeBaseMapper.selectList(
                new LambdaQueryWrapper<KnowledgeBase>()
                        .eq(KnowledgeBase::getUserId, userId)
                        .orderByDesc(KnowledgeBase::getCreatedAt)
        );
    }

    /**
     * 删除知识库，同时清理父块与向量。
     *
     * <p>注意：MySQL 的删除能跟着事务回滚，Redis 里的向量不能。
     * 所以顺序是"先删向量、再删表"，万一后面失败，至少不会留下查不到的脏数据。</p>
     */
    @DeleteMapping("/kb/delete")
    @Transactional(rollbackFor = Exception.class)
    public String deleteKnowledgeBase(@RequestParam String kbId,
                                      @RequestParam String userId) {
        KnowledgeBase kb = knowledgeBaseMapper.selectById(kbId);
        if (kb == null) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "知识库不存在");
        }
        if (!kb.getUserId().equals(userId)) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "无权删除该知识库");
        }

        // 1. 先清 Redis 里的子块（分批，避免一次性拉全量）
        int deletedVectors = vectorIndexService.deleteByFilter("kb_id == '" + kbId.replace("'", "") + "'");

        // 2. 再清 MySQL 里的父块
        parentBlockMapper.delete(new LambdaQueryWrapper<ParentBlock>()
                .eq(ParentBlock::getKbId, kbId)
                .eq(ParentBlock::getUserId, userId));

        // 3. 最后删知识库记录
        knowledgeBaseMapper.deleteById(kbId);

        log.info("知识库 {} 已删除，清理向量 {} 条", kbId, deletedVectors);
        return "删除成功，清理向量 " + deletedVectors + " 条";
    }
}
