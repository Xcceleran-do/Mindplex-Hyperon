# Cypher: create schema and indexes (idempotent)

CREATE CONSTRAINT content_unique IF NOT EXISTS
FOR (c:Content) REQUIRE c.contentId IS UNIQUE;

CREATE CONSTRAINT creator_unique IF NOT EXISTS
FOR (u:Creator) REQUIRE u.creatorId IS UNIQUE;

CREATE CONSTRAINT topic_unique IF NOT EXISTS
FOR (t:Topic) REQUIRE t.topicId IS UNIQUE;

CREATE CONSTRAINT tag_unique IF NOT EXISTS
FOR (g:Tag) REQUIRE g.tagId IS UNIQUE;

CREATE CONSTRAINT segment_unique IF NOT EXISTS
FOR (s:AudienceSegment) REQUIRE s.segmentId IS UNIQUE;

CREATE INDEX content_created_at IF NOT EXISTS FOR (c:Content) ON (c.createdAt);
CREATE INDEX content_format IF NOT EXISTS FOR (c:Content) ON (c.format);

RETURN 'schema_ok' AS status;
