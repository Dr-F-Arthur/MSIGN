import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from Bio import Align

'''
# 读取并严格清洗数据
df = pd.read_csv('Lppdb.csv', dtype={'smiles': str, 'new_split': str})
df = df.dropna(subset=['smiles', 'new_split'])
df['smiles'] = df['smiles'].str.strip()
df = df[df['smiles'] != '']

# 检查数据分割
train_df = df[df['category'].str.lower() == 'general']  # 兼容大小写
valid_df = df[df['category'].str.lower() == 'refined']  # 兼容大小写
print(f"有效训练集: {len(train_df)} | 有效验证集: {len(valid_df)}")

def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return GetMorganFingerprintAsBitVect(mol, 2, 2048)
    except:
        return None

# 处理训练集
train_fps = []
for s in tqdm(train_df['smiles'], desc="训练集处理"):
    fp = smiles_to_fingerprint(s)
    if fp: train_fps.append(fp)

# 处理验证集
valid_data = []
for pdb, s in tqdm(zip(valid_df['PDB Code'], valid_df['smiles']), desc="验证集处理"):
    fp = smiles_to_fingerprint(s)
    if fp: valid_data.append( (pdb, fp) )

# 计算相似度
results = []
for pdb, valid_fp in tqdm(valid_data, desc="相似度计算"):
    similarities = DataStructs.BulkTanimotoSimilarity(valid_fp, train_fps)
    results.append( {'PDB Code': pdb, 'Max Similarity': max(similarities) if similarities else 0.0} )

# 保存结果
if results:
    pd.DataFrame(results).to_csv('max_similarities_pdbbind.csv', index=False)
    print(f"成功保存 {len(results)} 条结果")
else:
    print("警告：无有效验证数据可保存")
'''
'''
df = pd.read_csv("Lppdb.csv")

# 分割训练集和验证集
train_df = df[df["category"] == "general"].dropna(subset=["sequence"])
valid_df = df[df["category"] == "refined"].dropna(subset=["sequence"])

# 提取序列和PDB编号
train_sequences = train_df["sequence"].tolist()
valid_sequences = valid_df["sequence"].tolist()
pdb_codes = valid_df["PDB Code"].tolist()

# 配置比对参数
aligner = Align.PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 1.0
aligner.mismatch_score = -1.0
aligner.gap_score = -2.0
aligner.extend_gap_score = -0.5

max_similarities = []

with tqdm(valid_sequences, desc="Processing", position=0) as main_pbar:
    for valid_seq in main_pbar:
        current_max = 0.0  # 初始值设为0，确保不会出现负数

        # 内层循环计算相似度（训练集超过1万条建议关闭内层进度条）
        for train_seq in train_sequences:
            score = aligner.score(valid_seq, train_seq)
            max_possible = len(valid_seq) * aligner.match_score
            similarity = max(score / max_possible, 0.0)  # 关键修改：限制在0-1之间

            if similarity > current_max:
                current_max = similarity
                main_pbar.set_postfix_str(f"Current Max: {current_max:.2f}")

        max_similarities.append(current_max)

# 保存结果
result_df = pd.DataFrame({
    "PDB Code": pdb_codes,
    "max_similarity": max_similarities
})
result_df.to_csv("max_seq_similarity_results.csv", index=False)
'''


# 读取数据并清洗
df = pd.read_csv('Lppdb.csv', dtype={'sequence': str})
df = df.dropna(subset=['sequence'])
df['sequence'] = df['sequence'].str.upper()  # 统一大写处理

# 数据集划分（假设refined为验证集）
train_df = df[df['new_split'] == 'train']
valid_df = df[df['new_split'] == 'val']

print(f"训练集样本: {len(train_df)}, 验证集样本: {len(valid_df)}")

# 准备数据
train_seqs = train_df['sequence'].tolist()
valid_data = list(zip(valid_df['PDB Code'], valid_df['sequence']))

def sequence_identity(seq1, seq2):
    """计算序列同一性（基于较短序列长度）"""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / min_len

# 计算最大相似度
results = []
for pdb, valid_seq in tqdm(valid_data, desc="处理验证集"):
    max_sim = 0.0
    for train_seq in train_seqs:
        sim = sequence_identity(valid_seq, train_seq)
        if sim > max_sim:
            max_sim = sim
            if max_sim == 1.0:  # 发现完全匹配时提前终止
                break
    results.append({'PDB Code': pdb, 'Max Similarity': max_sim})

# 保存结果
result_df = pd.DataFrame(results)
result_df.to_csv('protein_max_similarity_lppdb.csv', index=False)
print(f"完成！有效结果数量：{len(result_df)}")