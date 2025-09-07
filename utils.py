import logging
import os
import matplotlib.pyplot as plt
import json
import random

def get_random_character(purpose=True):

    # 1.人口统计特征（Demographic Characteristics）
    age_roles = [
        "aged between 18 and 24",
        "aged between 25 and 34",
        "aged between 35 and 44",
        "aged between 45 and 54",
        "aged between 55 and 64",
        "aged 65 or older"
    ]

    gender_roles = [
        "male",
        "female",
        "non-binary or other gender"
    ]

    income_roles = [
        "with a low income level",
        "with a middle income level",
        "with a high income level"
    ]

    # 2. 社会经济特征（Socioeconomic Characteristics）
    occupation_roles = [
        "a student",
        "an employee",
        "self-employed",
        "retired",
        "unemployed"
    ]

    education_roles = [
        "with a high school education",
        "with an associate degree",
        "with a bachelor's degree",
        "with a master's degree",
        "with a doctorate"
    ]

    # 3. 行为和态度特征（Behavioral and Attitudinal Characteristics）：
    risk_preference_roles = [
        "risk-averse",
        "risk-neutral",
        "risk-seeking"
    ]

    # 实际出行需求（Actual Travel Needs）：
    travel_purpose_roles = [
        "traveling for commuting",
        "traveling for shopping",
        "traveling for leisure",
        "traveling for business",
        "traveling for education"
    ]
    if purpose:
        character_description = (f"You are a {random.choice(gender_roles)} character, {random.choice(age_roles)}, "
                             f"{random.choice(income_roles)}, {random.choice(occupation_roles)}, "
                             f"{random.choice(education_roles)}, {random.choice(risk_preference_roles)}, and "
                             f"{random.choice(travel_purpose_roles)}.")
    else:  # no purpose
        character_description = (f"You are a {random.choice(gender_roles)} character, {random.choice(age_roles)}, "
                                 f"{random.choice(income_roles)}, {random.choice(occupation_roles)}, "
                                 f"{random.choice(education_roles)}, and {random.choice(risk_preference_roles)}.")
    return character_description

def create_output_directory(base_dir='output', prefix='run_'):
    """
    Create a new output directory with the format base_dir/prefix_i, where i is an incremented integer.
    If base_dir does not exist, it will be created.

    Args:
        base_dir (str): The base directory where the output directories will be created.
        prefix (str): The prefix for the output directories.

    Returns:
        str: The path to the new output directory.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")

    # Find the maximum value of i in existing prefix_i directories
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(prefix) and d[len(prefix):].isdigit()]
    max_index = -1
    for d in existing_dirs:
        try:
            i = int(d[len(prefix):])
            if i > max_index:
                max_index = i
        except ValueError:
            continue

    # New directory index
    new_index = max_index + 1
    new_dir = os.path.join(base_dir, f'{prefix}{new_index}')
    os.makedirs(new_dir)
    print(f"Created new directory: {new_dir}")

    return new_dir


def setup_logging(log_file_path):
    """
    Set up logging configuration to log messages to a specified file and to the console.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def append_to_file(value, filename='output.txt'):
    with open(filename, 'a') as file:
        file.write(f"{value}\n")

def read_values(filename='output.txt'):
    with open(filename, 'r') as file:
        values = [float(line.strip()) for line in file]
    return values

def save_dict_to_json(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=4, default=str)



if __name__ == '__main__':

    # 使用示例
    new_output_dir = create_output_directory()
    log_file = os.path.join(new_output_dir, 'test.log')
    logger = setup_logging(log_file)

    # 记录示例日志
    logger.debug("This is debug")
    logger.info("This is info")
    logger.error("This is error")

    print(f"Log file created at: {log_file}")

    # 模拟迭代过程
    for i in range(10):
        # 假设这里计算出一个值
        value = i ** 2
        append_to_file(value)

    # 读取文件中的值
    values = read_values()

    # 绘图
    plt.plot(values)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Iteration Values')
    plt.show()
