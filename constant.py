
PRETRAINED_MODELS = {
    # 'codegen-350m': '/home/public_space/yanmeng/zhangjingrui/models/codegen-350M-multi',
    'starcoder-7b':'/home/public_space/liuchao/shushanfu/LMOps/checkpoints/StarCoder-7B',
    'starcoder-1b':'/home/public_space/liuchao/shushanfu/LMOps/checkpoints/StarCoder-1B',
    'deepseek-coder-1.3b':'/home/public_space/yanmeng/zhangjingrui/models/deepseek-coder-1.3b-base',
    'deepseek-coder-6.7b':'/home/public_space/yanmeng/zhangjingrui/models/deepseek-coder-6.7b-base',
    'qwen2.5-coder-0.5B':'/home/public_space/yanmeng/lidong/models/Qwen2.5-Coder-0.5B-Instruct',
    'qwen2.5-coder-3B':'/home/public_space/yanmeng/zhangjingrui/models/Qwen2.5-Coder-3B',
    'qwen2.5-coder-7B':'/home/public_space/yanmeng/zhangjingrui/models/Qwen2.5-Coder-7B',
}

SECURITY_MODELS = {
'qwen2.5-coder-0.5B': '/home/public_space/yanmeng/zhangjingrui/models/checkpoint-last',
}

ALL_VUL_TYPES = ['cwe-089', 'cwe-125', 'cwe-078', 'cwe-476', 'cwe-416', 'cwe-022', 'cwe-787', 'cwe-079', 'cwe-190']


VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-022', '2-py'),
    ('cwe-787', '2-c'),
}

TEST_SCENARIOS = [
    ['cwe-089', '0-py', '1-py'],
    ['cwe-125', '0-c', '1-c'],
    ['cwe-078', '0-py', '1-py'],
    ['cwe-476', '0-c', '2-c'],
    ['cwe-416', '0-c'],
    ['cwe-022', '0-py', '1-py'],
    ['cwe-787', '0-c', '1-c'],
    ['cwe-079', '0-py', '1-py'],
    ['cwe-190', '0-c', '1-c'],
    ['cwe-416', '1-c'],
]

# NOTTRAINED_VUL_TYPES = ['cwe-119', 'cwe-502', 'cwe-732', 'cwe-798']
NOTTRAINED_VUL_TYPES = ['cwe-119', 'cwe-502', 'cwe-732', 'cwe-798','cwe-020', 'cwe-117', 'cwe-777']

DOP_VUL_TYPES = ['cwe-089']

DOP_SCENARIOS = [
    ('cwe-089', 'con'),
    ('cwe-089', 'm-1'),
    ('cwe-089', 'm-2'),
    ('cwe-089', 'm-3'),
    ('cwe-089', 'm-4'),
    ('cwe-089', 'd-1'),
    ('cwe-089', 'd-2'),
    ('cwe-089', 'd-3'),
    ('cwe-089', 'd-4'),
    ('cwe-089', 'd-5'),
    ('cwe-089', 'd-6'),
    ('cwe-089', 'd-7'),
    ('cwe-089', 'c-1'),
    ('cwe-089', 'c-2'),
    ('cwe-089', 'c-3'),
    ('cwe-089', 'c-4'),
    ('cwe-089', 'c-5'),
]