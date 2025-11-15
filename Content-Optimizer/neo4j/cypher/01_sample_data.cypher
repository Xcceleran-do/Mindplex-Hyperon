// Sample data: creates a few creators, content nodes, topics and audience segments with ENGAGED_WITH edges

CREATE (u1:Creator {creatorId: 'creator_1', name: 'Alice', createdAt: datetime()});
CREATE (u2:Creator {creatorId: 'creator_2', name: 'Bob', createdAt: datetime()});

CREATE (t1:Topic {topicId: 'topic_ai', name: 'AI'});
CREATE (t2:Topic {topicId: 'topic_music', name: 'Music'});

CREATE (c1:Content {contentId: 'c1', title: 'Intro to AI', lengthSec: 300, format: 'video', platform: 'youtube', createdAt: datetime()});
CREATE (c2:Content {contentId: 'c2', title: 'AI for Musicians', lengthSec: 420, format: 'article', platform: 'blog', createdAt: datetime()});
CREATE (c3:Content {contentId: 'c3', title: 'Music Theory Basics', lengthSec: 180, format: 'short', platform: 'instagram', createdAt: datetime()});
CREATE (c4:Content {contentId: 'c4', title: 'Advanced ML', lengthSec: 900, format: 'video', platform: 'youtube', createdAt: datetime()});
CREATE (c5:Content {contentId: 'c5', title: 'Songwriting Tips', lengthSec: 240, format: 'article', platform: 'blog', createdAt: datetime()});
CREATE (c6:Content {contentId: 'c6', title: 'Generative Music', lengthSec: 360, format: 'video', platform: 'youtube', createdAt: datetime()});

// relationships
CREATE (u1)-[:CREATED]->(c1);
CREATE (u1)-[:CREATED]->(c2);
CREATE (u2)-[:CREATED]->(c3);
CREATE (u2)-[:CREATED]->(c4);

CREATE (c1)-[:HAS_TOPIC]->(t1);
CREATE (c2)-[:HAS_TOPIC]->(t1);
CREATE (c3)-[:HAS_TOPIC]->(t2);
CREATE (c5)-[:HAS_TOPIC]->(t2);

CREATE (s1:AudienceSegment {segmentId: 'seg_1', name: 'EngagedUsers'});
CREATE (s2:AudienceSegment {segmentId: 'seg_2', name: 'CasualViewers'});

CREATE (s1)-[:ENGAGED_WITH {views: 100, likes: 10, watch_time: 25000}]->(c1);
CREATE (s1)-[:ENGAGED_WITH {views: 50, likes: 5, watch_time: 10000}]->(c2);
CREATE (s2)-[:ENGAGED_WITH {views: 30, likes: 2, watch_time: 4000}]->(c3);

RETURN 'sample_data_inserted' AS status;
