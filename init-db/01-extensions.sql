-- Runs once on first container start (docker-entrypoint-initdb.d).
-- pgvector backs the dense half of hybrid retrieval; the sparse half uses
-- the built-in tsvector/GIN, which needs no extension.
CREATE EXTENSION IF NOT EXISTS vector;
