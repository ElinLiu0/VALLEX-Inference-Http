from datetime import datetime,timedelta,timezone
from typing import *
 
def StaticShiftingTimeStampValidation(
    allowedShiftedTime: float,
    InputStrTime: str
):
    """
    静态偏移时间戳验证函数

    参数:
        allowedShiftedTime (float): 允许的时间偏移量（以秒为单位）
        InputStrTime (str): 输入的时间字符串（格式为'%Y/%m/%d %H:%M:%S'）

    返回值:
        bool: 如果输入时间戳在允许的时间范围内，则返回True；否则返回False
    """
    # 获取服务器得到来自终端请求时时间(UTC时间)
    current = datetime.now(tz=timezone.utc)
    InputStrTime = datetime.strptime(InputStrTime, '%Y/%m/%d %H:%M:%S')
    InputTimeStamp = InputStrTime.timestamp()
    # 将允许的时间偏移量转换为时间间隔
    allowedShiftedTime = timedelta(seconds=allowedShiftedTime)
    # 计算基于当前时间戳向前偏移的时间戳
    shiftedTime = current - allowedShiftedTime
    shiftedTime = shiftedTime.timestamp()
    # 将当前服务器的响应时间转换为时间戳
    current = current.timestamp()
    # 判断当前时间戳是否在允许的时间范围内
    if InputTimeStamp >= shiftedTime and InputTimeStamp <= current:
        return True
    else:
        return False
