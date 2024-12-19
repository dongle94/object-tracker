import random


def get_color_for_id(tracking_id, seed=42):
    """
    특정 ID를 고유한 색상(BGR)으로 매핑하는 함수.

    Args:
        tracking_id (int): 고유한 정수형 ID.
        seed (int, optional): 색상 매핑의 일관성을 위한 난수 시드. 기본값은 42.

    Returns:
        tuple: OpenCV에서 사용하는 BGR 색상 (R, G, B).
    """
    random.seed(tracking_id + seed)  # 고유한 seed 기반으로 난수 생성
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return b, g, r  # OpenCV에서 사용하는 BGR 형식
