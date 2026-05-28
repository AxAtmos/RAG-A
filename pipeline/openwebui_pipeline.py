import requests


class Pipeline:
    id = "rag_knowledge_base"
    name = ""
    description = "基于企业知识库的 RAG 问答系统"
    type = "manifold"

    def __init__(self):
        self.api_url = "http://192.168.10.104:8000/api/query-sync"
        self.timeout = 120

    async def on_startup(self):
        print(f"[RAG Pipeline] Ready. API: {self.api_url}")

    async def on_shutdown(self):
        pass

    def pipelines(self):
        return [
            {
                "id": "rag_knowledge_base",
                "name": "RAG 知识库问答",
                "description": "从企业知识库中检索文档并回答",
            }
        ]

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict) -> str:
        question = user_message
        if not question.strip():
            return "请输入问题"

        try:
            resp = requests.post(
                self.api_url,
                json={"question": question},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("answer", "未能获取到回答")
            sources = data.get("sources", [])
            confidence = data.get("confidence", "")

            if sources:
                files = list(dict.fromkeys(
                    s.get("file_name", "") for s in sources if s.get("file_name")
                ))
                if files:
                    answer += f"\n\n---\n**引用来源：** {', '.join(files)}"
                    if confidence:
                        label = "高置信" if confidence == "high" else "低置信"
                        answer += f"  [{label}]"

            return answer

        except requests.Timeout:
            return "请求超时，请稍后重试"
        except requests.ConnectionError:
            return "无法连接到 RAG API 服务"
        except Exception as e:
            return f"请求失败：{str(e)}"
