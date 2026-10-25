package com.example.rag.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.rag.entity.ParentBlock;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface ParentBlockMapper extends BaseMapper<ParentBlock> {

    /**
     * 关键词召回：对父块正文做全文检索（数据库侧使用 ngram 分词，见 docs/schema.sql）。
     *
     * <p>这是多路召回中的"文本路"：向量召回擅长语义相近，但对错误码、类名、
     * 专有名词这类精确匹配不敏感；全文索引正好补上这一点。</p>
     *
     * @param keyword 已经清洗过的检索词（空格分隔，已去掉 + - &gt; ( ) 等布尔语法符号）
     */
    @Select("""
            SELECT parent_id
            FROM rag_parent_block
            WHERE user_id = #{userId}
              AND kb_id = #{kbId}
              AND MATCH(content) AGAINST(#{keyword} IN BOOLEAN MODE)
            ORDER BY MATCH(content) AGAINST(#{keyword} IN BOOLEAN MODE) DESC
            LIMIT #{limit}
            """)
    List<String> searchByKeyword(@Param("userId") String userId,
                                 @Param("kbId") String kbId,
                                 @Param("keyword") String keyword,
                                 @Param("limit") int limit);

    /**
     * 关键词召回的兜底路径：整句走自然语言模式。
     *
     * <p>BOOLEAN 模式下 ngram 会要求一个词的所有 2-gram 都命中，
     * 问句里一旦带上「的内容是什么」这类口语成分，相邻 2-gram 在原文中不存在，
     * 整句就会返回 0 行。此时退化成自然语言模式，靠相关度排序兜底。</p>
     */
    @Select("""
            SELECT parent_id
            FROM rag_parent_block
            WHERE user_id = #{userId}
              AND kb_id = #{kbId}
              AND MATCH(content) AGAINST(#{question} IN NATURAL LANGUAGE MODE)
            ORDER BY MATCH(content) AGAINST(#{question} IN NATURAL LANGUAGE MODE) DESC
            LIMIT #{limit}
            """)
    List<String> searchByQuestionNatural(@Param("userId") String userId,
                                         @Param("kbId") String kbId,
                                         @Param("question") String question,
                                         @Param("limit") int limit);
}
