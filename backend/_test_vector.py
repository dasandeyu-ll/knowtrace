"""验证向量检索：用不含原文关键词的改写问法，检查能否语义命中正确段落"""
import os
from vector_store import VectorStore, chunk_text

DOC = """第一段：差旅住宿标准
员工出差到北京、上海、广州、深圳，酒店开销每天最高五百元，餐补每天一百二十元。
前往成都、武汉、西安等二线城市，酒店开销每天最高三百五十元，餐补每天一百元。

第二段：交通工具选择
市内交通凭票实报实销。跨城出行优先选择高铁二等座；若航程超过三小时，可申请经济舱机票。

第三段：报销时效
费用发生后七个工作日内必须提交申请，并附发票原件与消费明细。部门负责人审批后，
财务部在三个工作日内打款。单笔金额超过五千元的，需额外经总经理签批。

第四段：客户招待
招待客户每人每次不超过二百元，需提前报备招待对象与事由。发票须备注用餐人数。

第五段：违规责任
虚报或伪造发票，首次警告并追回款项，第二次记过并取消当年评优资格，情节严重的解除劳动合同。
"""

def main():
    chunks_path = os.path.join(os.path.dirname(__file__), "_test_chunks.json")
    index_path = os.path.join(os.path.dirname(__file__), "_test_faiss.index")
    for p in (chunks_path, index_path):
        if os.path.exists(p):
            os.remove(p)

    # 按段落独立入库，保证 ground truth 唯一
    segs = [s.strip() for s in DOC.split("第") if s.strip()]
    paras = []
    for s in segs:
        body = s.split("\n", 1)[1].replace("\n", "") if "\n" in s else s
        title = s.split("\n", 1)[0]
        paras.append((title, body))
    store = VectorStore(index_path, chunks_path)
    store.add([{"text": body, "source": f"{title}.txt", "chunk": i} for i, (title, body) in enumerate(paras)])
    store.build_index()
    print(f"mode={store.mode} chunks={len(store.chunks)} dim={store._dim}")
    store.save()

    store2 = VectorStore(index_path, chunks_path)
    print(f"reload mode={store2.mode} ntotal={None if store2.index is None else store2.index.ntotal}")

    cases = [
        ("住酒店一天最多能报多少钱", "差旅住宿标准"),
        ("去外地开会怎么坐车", "交通工具选择"),
        ("发票最晚什么时候交给财务", "报销时效"),
        ("请客户吃饭有什么限额", "客户招待"),
        ("报销作假会怎么样", "违规责任"),
    ]
    ok = 0
    for q, expect in cases:
        hits = store2.search(q, k=1)
        hit = hits[0]["source"].replace(".txt", "") if hits else "?"
        good = expect in hit
        ok += good
        print(f"[{'OK ' if good else 'MISS'}] {q}  ->  {hit}")
    print(f"\n语义命中 {ok}/{len(cases)}")

    for p in (chunks_path, index_path):
        if os.path.exists(p):
            os.remove(p)

if __name__ == "__main__":
    main()
