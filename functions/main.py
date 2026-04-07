from firebase_functions import https_fn
from firebase_admin import initialize_app, firestore
import math
import json

# 1. 初始化
initialize_app()

@https_fn.on_request()
def get_course_recommendation(req: https_fn.Request) -> https_fn.Response:
    user_email = req.args.get("email")
    if not user_email:
        return https_fn.Response("Missing email parameter", status=400)

    db = firestore.client()

    try:
        # 2. 抓使用者興趣
        user_docs = db.collection("users").where("email", "==", user_email).limit(1).get()
        if not user_docs:
            return https_fn.Response("User not found", status=404)
        
        user_data = user_docs[0].to_dict()
        user_interests = user_data.get("interests", [])
        user_name = user_data.get("name", "Student")

        tag_map = {
            "AI": "ai_algo",
            "商業管理": "biz_mgt",
            "資料分析": "data_ana",
            "ERP系統": "erp_sys",
            "軟體開發": "soft_dev",
            "系統架構": "sys_infra"
        }

        user_vec = {v: 0.0 for v in tag_map.values()}
        for interest in user_interests:
            if interest in tag_map:
                user_vec[tag_map[interest]] = 1.0

        # 3. 算 KNN 距離
        courses_ref = db.collection("courses").stream()
        results = []

        for doc in courses_ref:
            course_data = doc.to_dict()
            weights = course_data.get("ai_weights", {})
            
            display_name = course_data.get("course_name", doc.id)
            c_type = course_data.get("course_type", "必修")

            dist_sq = 0
            for field in tag_map.values():
                u_val = user_vec.get(field, 0.0)
                c_val = weights.get(field, 0.0)
                dist_sq += (u_val - c_val) ** 2
            
            distance = math.sqrt(dist_sq)
            results.append({
                "course_name": display_name,
                "course_type": c_type,
                "distance": round(distance, 4)
            })

        results.sort(key=lambda x: x["distance"])

        # 4. 回傳結果
        response_body = {
            "user": user_name,
            "interest_tags": user_interests,
            "recommendations": results[:5]
        }
        
        return https_fn.Response(
            json.dumps(response_body, ensure_ascii=False), 
            mimetype="application/json"
        )

    except Exception as e:
        # 這裡就是剛才報錯的地方，確保括號都有對齊
        return https_fn.Response(f"Error: {str(e)}", status=500)