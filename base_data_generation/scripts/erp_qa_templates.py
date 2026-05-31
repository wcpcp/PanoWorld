from __future__ import annotations

from typing import Any


QA_TEMPLATES: dict[str, dict[str, Any]] = {
    "1_basic_understanding": {
        "1.1_grounding_to_text": {
            "instruction": "根据坐标定位到 ERP 中的目标并识别语义。",
            "prompts": [
                "在整张 ERP 全景图中，坐标 <box> {bbox} </box> 对应的物体是什么？请给出简短描述。",
                "请识别 ERP 图中 <box> {bbox} </box> 的目标，并总结可见属性。",
            ],
            "answer_format": "{identify}。{caption_brief}",
        },
        "1.1_text_to_grounding": {
            "instruction": "根据文本指代，在 ERP 中定位目标。",
            "prompts": [
                "请在 ERP 全景图中找出 {unique_description}，并输出坐标框。",
                "定位场景中的 {unique_description}，返回其包围盒坐标。",
            ],
            "answer_format": "<box> {bbox} </box>",
        },
        "1.2_existence": {
            "instruction": "判断某类别在 ERP 场景中是否存在。",
            "prompts": [
                "在这张 ERP 全景图中能看到 {category} 吗？",
                "该全景场景是否包含 {category}？",
            ],
            "answer_format": "{yes_no}。",
        },
        "1.2_counting": {
            "instruction": "统计 ERP 场景中的可数目标。",
            "prompts": [
                "ERP 场景中可见多少个 {category}？",
                "请统计这张全景图里 {category} 的数量。",
            ],
            "answer_format": "图中可见 {count} 个{category}。",
        },
    },
    "2_omnidirectional_understanding": {
        "2.1_directive_direction": {
            "instruction": "输出目标相对 ERP 正前方向的绝对方位。",
            "prompts": [
                "以 ERP 画面中心对应朝向为正前方，<box> {bbox} </box> 在什么方位？",
                "相对于初始朝向（yaw=0），{unique_description} 位于哪个方向？",
            ],
            "answer_format": "它位于 {direction_8}（约 {yaw_angle} 度，约 {clock_direction}）。",
        },
        "2.2_relative_direction": {
            "instruction": "判断两个目标在球面空间中的相对方向。",
            "prompts": [
                "在 ERP 的球面空间中，{unique_description_A} 相对 {unique_description_B} 在什么方向？",
                "比较 <box> {bbox_A} </box> 与 <box> {bbox_B} </box>，A 相对 B 的方位是什么？",
            ],
            "answer_format": "{unique_description_A} 在 {unique_description_B} 的 {relative_position}。",
        },
        "2.3_boundary_continuity": {
            "instruction": "识别 ERP 左右边界的连通目标。",
            "prompts": [
                "ERP 图像左右边界是连通的。请指出一个在边界处被切断但实际连续的目标。",
                "在全景图左右两端，哪个目标是同一个实体的跨边界呈现？",
            ],
            "answer_format": "跨边界目标为 <box> {bbox_boundary} </box>（{identify}）。",
        },
        "2.4_distortion_awareness": {
            "instruction": "解释极区拉伸与真实形状的差异。",
            "prompts": [
                "位于极区的 <box> {bbox} </box> 在 ERP 中发生拉伸。请说明其真实形状特征。",
            ],
            "answer_format": "{distortion_explanation}",
        },
    },
    "3_3d_spatial_understanding": {
        "3.1_distance_estimation": {
            "instruction": "根据深度统计估计目标距离。",
            "prompts": [
                "请估计 ERP 中 <box> {bbox} </box> 到相机的大致距离。",
                "目标 {unique_description} 的物理距离大约是多少？",
            ],
            "answer_format": "大约 {distance_bucket}（约 {distance_m:.2f} 米）。",
        },
        "3.2_relative_positioning": {
            "instruction": "比较两个目标在 3D 中的远近。",
            "prompts": [
                "<box> {bbox_A} </box> 和 <box> {bbox_B} </box> 哪个离相机更近？",
                "比较 {unique_description_A} 与 {unique_description_B}，谁距离观察者更近？",
            ],
            "answer_format": "{closer_item} 更近（A 约 {distance_A:.2f} 米，B 约 {distance_B:.2f} 米）。",
        },
    },
}


NEGATIVE_EXISTENCE_CANDIDATES: list[str] = [
    "冰箱",
    "微波炉",
    "洗衣机",
    "浴缸",
    "钢琴",
    "自行车",
    "烤箱",
    "投影仪",
]


COUNTABLE_CATEGORIES_HINT: list[str] = [
    "椅子",
    "凳子",
    "桌子",
    "沙发",
    "门",
    "窗",
    "灯",
    "柜子",
    "屏幕",
    "电视",
]


OPEN_TASK_PROMPTS: dict[str, str] = {
    "scene_layout": (
        "你是 ERP 全景理解数据生成器。输入是一张完整 ERP 全景图。"
        "请只基于图像可见信息，生成一个 JSON 对象，格式为: "
        "{\"qas\":[{\"task\":\"scene_layout\",\"question\":string,\"answer\":string}]}。"
        "要求: 只输出 JSON；question 聚焦场景类型与布局；answer 简洁客观，不编造不可见信息。"
    ),
    "polar_distortion": (
        "你是 ERP 极区畸变理解数据生成器。输入是一张 ERP 全景图和目标框描述。"
        "请生成一个 JSON 对象: "
        "{\"qas\":[{\"task\":\"polar_distortion\",\"question\":string,\"answer\":string}]}。"
        "要求: question 必须指向给定目标；answer 解释 ERP 拉伸现象与真实形状差异；只输出 JSON。"
    ),
}
