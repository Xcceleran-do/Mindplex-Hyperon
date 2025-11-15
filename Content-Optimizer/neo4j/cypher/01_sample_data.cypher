// Sample data (idempotent): MERGE nodes by their unique ids and set properties.

MERGE (u1:Creator {creatorId: 'creator_1'})
SET u1.name = 'Alice', u1.createdAt = coalesce(u1.createdAt, datetime());

MERGE (u2:Creator {creatorId: 'creator_2'})
SET u2.name = 'Bob', u2.createdAt = coalesce(u2.createdAt, datetime());

MERGE (t1:Topic {topicId: 'topic_ai'})
SET t1.name = 'AI';

MERGE (t2:Topic {topicId: 'topic_music'})
SET t2.name = 'Music';

MERGE (c1:Content {contentId: 'c1'})
SET c1.title = 'Intro to AI', c1.lengthSec = 300, c1.format = 'video', c1.platform = 'youtube', c1.createdAt = coalesce(c1.createdAt, datetime());

MERGE (c2:Content {contentId: 'c2'})
SET c2.title = 'AI for Musicians', c2.lengthSec = 420, c2.format = 'article', c2.platform = 'blog', c2.createdAt = coalesce(c2.createdAt, datetime());

MERGE (c3:Content {contentId: 'c3'})
SET c3.title = 'Music Theory Basics', c3.lengthSec = 180, c3.format = 'short', c3.platform = 'instagram', c3.createdAt = coalesce(c3.createdAt, datetime());

MERGE (c4:Content {contentId: 'c4'})
SET c4.title = 'Advanced ML', c4.lengthSec = 900, c4.format = 'video', c4.platform = 'youtube', c4.createdAt = coalesce(c4.createdAt, datetime());

MERGE (c5:Content {contentId: 'c5'})
SET c5.title = 'Songwriting Tips', c5.lengthSec = 240, c5.format = 'article', c5.platform = 'blog', c5.createdAt = coalesce(c5.createdAt, datetime());

MERGE (c6:Content {contentId: 'c6'})
SET c6.title = 'Generative Music', c6.lengthSec = 360, c6.format = 'video', c6.platform = 'youtube', c6.createdAt = coalesce(c6.createdAt, datetime());

// relationships (idempotent)
MERGE (u1)-[:CREATED]->(c1);
MERGE (u1)-[:CREATED]->(c2);
MERGE (u2)-[:CREATED]->(c3);
MERGE (u2)-[:CREATED]->(c4);

MERGE (c1)-[:HAS_TOPIC]->(t1);
MERGE (c2)-[:HAS_TOPIC]->(t1);
MERGE (c3)-[:HAS_TOPIC]->(t2);
MERGE (c5)-[:HAS_TOPIC]->(t2);

MERGE (s1:AudienceSegment {segmentId: 'seg_1'})
SET s1.name = 'EngagedUsers';

MERGE (s2:AudienceSegment {segmentId: 'seg_2'})
SET s2.name = 'CasualViewers';

// ENGAGED_WITH edges with properties and timestamp
MERGE (s1)-[e1:ENGAGED_WITH]->(c1)
SET e1.views = 100, e1.likes = 10, e1.watch_time = 25000, e1.timestamp = coalesce(e1.timestamp, datetime());

MERGE (s1)-[e2:ENGAGED_WITH]->(c2)
SET e2.views = 50, e2.likes = 5, e2.watch_time = 10000, e2.timestamp = coalesce(e2.timestamp, datetime());

MERGE (s2)-[e3:ENGAGED_WITH]->(c3)
SET e3.views = 30, e3.likes = 2, e3.watch_time = 4000, e3.timestamp = coalesce(e3.timestamp, datetime());

RETURN 'sample_data_upserted' AS status;
