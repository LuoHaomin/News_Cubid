"""
相似度计算工具模块
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from typing import Union


def cosine_similarity_sparse(matrix: csr_matrix) -> csr_matrix:
    """
    计算稀疏矩阵的余弦相似度
    
    Args:
        matrix: 稀疏矩阵（用户-物品交互矩阵或用户-用户交互矩阵）
        
    Returns:
        余弦相似度矩阵（稀疏矩阵），值在[0, 1]范围内
    """
    # 归一化矩阵
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1))
    norms = np.array(norms).flatten()
    norms[norms == 0] = 1  # 避免除零
    
    # 计算归一化后的点积
    normalized_matrix = matrix.multiply(1.0 / norms[:, np.newaxis])
    similarity = normalized_matrix.dot(normalized_matrix.T)
    
    # 余弦相似度范围是[-1, 1]，归一化到[0, 1]
    # 使用clip确保值在[0, 1]范围内（处理浮点数精度问题）
    similarity.data = np.clip((similarity.data + 1) / 2, 0.0, 1.0)
    
    return similarity


def cosine_similarity_dense(vectors: np.ndarray) -> np.ndarray:
    """
    计算密集向量的余弦相似度矩阵
    
    Args:
        vectors: 向量矩阵，形状为(n, dim)
        
    Returns:
        余弦相似度矩阵，形状为(n, n)，值在[0, 1]范围内
    """
    # 归一化向量
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized_vectors = vectors / norms
    
    # 计算余弦相似度（点积）
    similarity = np.dot(normalized_vectors, normalized_vectors.T)
    
    # 确保值在[0, 1]范围内（余弦相似度范围是[-1, 1]，这里归一化到[0, 1]）
    # 使用clip确保值在[0, 1]范围内（处理浮点数精度问题）
    similarity = np.clip((similarity + 1) / 2, 0.0, 1.0)
    
    return similarity


def jaccard_similarity(matrix: csr_matrix) -> csr_matrix:
    """
    计算稀疏矩阵的Jaccard相似度
    
    Args:
        matrix: 稀疏矩阵（二进制矩阵）
        
    Returns:
        Jaccard相似度矩阵（稀疏矩阵）
    """
    # 转换为二进制矩阵
    binary_matrix = matrix.astype(bool).astype(int)
    
    # 计算交集和并集
    intersection = binary_matrix.dot(binary_matrix.T)
    row_sums = binary_matrix.sum(axis=1).A1
    union = row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - intersection
    
    # 避免除零
    union[union == 0] = 1
    similarity = intersection / union
    
    return csr_matrix(similarity)


def validate_similarity_matrix(matrix: Union[csr_matrix, np.ndarray], 
                              expected_shape: tuple = None,
                              tolerance: float = 1e-10) -> None:
    """
    验证相似度矩阵
    
    Args:
        matrix: 相似度矩阵
        expected_shape: 期望的形状（可选）
        tolerance: 浮点数容差，默认1e-10
        
    Raises:
        ValueError: 如果矩阵无效
    """
    if isinstance(matrix, csr_matrix):
        shape = matrix.shape
        if shape[0] != shape[1]:
            raise ValueError(f"Similarity matrix must be square, got {shape}")
        # 检查值范围（允许微小的浮点数误差）
        min_val = matrix.data.min()
        max_val = matrix.data.max()
        if max_val > 1.0 + tolerance or min_val < 0.0 - tolerance:
            raise ValueError(f"Similarity values must be in [0, 1], got [{min_val}, {max_val}]")
    elif isinstance(matrix, np.ndarray):
        shape = matrix.shape
        if shape[0] != shape[1]:
            raise ValueError(f"Similarity matrix must be square, got {shape}")
        # 检查值范围（允许微小的浮点数误差）
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val > 1.0 + tolerance or min_val < 0.0 - tolerance:
            raise ValueError(f"Similarity values must be in [0, 1], got [{min_val}, {max_val}]")
    else:
        raise ValueError(f"Invalid matrix type: {type(matrix)}")
    
    if expected_shape and shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {shape}")

