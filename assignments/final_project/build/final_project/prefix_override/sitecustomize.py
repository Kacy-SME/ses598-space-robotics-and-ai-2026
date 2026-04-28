import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/ses598-space-robotics-and-ai-2026/assignments/final_project/install/final_project'
