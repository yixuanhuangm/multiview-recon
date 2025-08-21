import sqlite3
import os

# 数据库路径
db_path = "database.db"  # 替换成你的 database.db 路径

if not os.path.exists(db_path):
    print("数据库文件不存在:", db_path)
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询每张图像的特征点数量
cursor.execute("""
    SELECT name, rows
    FROM images
    JOIN (SELECT image_id, COUNT(*) AS rows FROM keypoints GROUP BY image_id) AS kp
    ON images.image_id = kp.image_id
""")
features = cursor.fetchall()

print("每张图像的特征点数量：")
for name, count in features:
    print(f"{name}: {count} 特征点")

# 查询每张图像匹配到多少张其他图片
cursor.execute("""
    SELECT im1.name, COUNT(DISTINCT im2.id) as matched_images
    FROM matches AS m
    JOIN images AS im1 ON m.image_id1 = im1.image_id
    JOIN images AS im2 ON m.image_id2 = im2.image_id
    GROUP BY im1.name
""")
matches = cursor.fetchall()

print("\n每张图像匹配到的其他图片数：")
for name, matched_count in matches:
    print(f"{name}: {matched_count} 张图片匹配")
    
conn.close()
