-- 企业知识库 RAG 问答系统 —— 数据库结构
-- 用法：mysql -uroot -p < docs/schema.sql
-- Docker 方式：首次 docker compose up 时自动执行

CREATE DATABASE IF NOT EXISTS rag_kb_db
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE rag_kb_db;

DROP TABLE IF EXISTS rag_parent_block;
DROP TABLE IF EXISTS knowledge_base;

-- 知识库
CREATE TABLE knowledge_base
(
    id         VARCHAR(64)  NOT NULL COMMENT '知识库ID（UUID）',
    user_id    VARCHAR(64)  NOT NULL COMMENT '所属用户',
    name       VARCHAR(128) NOT NULL COMMENT '知识库名称',
    created_at DATETIME     NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_user (user_id)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='知识库';

-- 父块：比子块大，最终喂给模型的是它
CREATE TABLE rag_parent_block
(
    id          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '主键',
    parent_id   VARCHAR(160) NOT NULL COMMENT '父块ID，子块通过它回查',
    source      VARCHAR(255) NOT NULL COMMENT '来源文件名，手动录入时为 manual_input',
    content     LONGTEXT COMMENT '父块正文',
    user_id     VARCHAR(64)  NOT NULL COMMENT '所属用户',
    kb_id       VARCHAR(64)  NOT NULL COMMENT '所属知识库',
    chunk_index INT          NOT NULL DEFAULT 0 COMMENT '块序号',
    created_at  DATETIME     NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_parent (parent_id),
    KEY idx_user_kb (user_id, kb_id),
    -- ngram 分词，让中文也能走全文索引（MySQL 5.7.6+ 内置）
    FULLTEXT KEY ft_content (content) WITH PARSER ngram
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci COMMENT ='父块（检索与上下文单位）';
