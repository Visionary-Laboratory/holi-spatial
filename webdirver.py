from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

# 配置浏览器
chrome_options = Options()
# 强制不使用代理
chrome_options.add_argument('--proxy-server="direct://"')
chrome_options.add_argument('--proxy-bypass-list=*')
# chrome_options.add_argument("--headless")  # 如果想看运行过程，请注释掉这行
chrome_options.add_argument("--window-size=375,812")  # 模拟手机屏幕尺寸
chrome_options.add_argument("user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def extract_report():
    try:
        driver.get("https://activity.wifiwx.com/webapp/2025/reading-report/#/main")
        wait = WebDriverWait(driver, 10)

        # 1. 处理可能的登录或启动封面（需要根据页面实际按钮的 class 调整）
        print("等待页面加载...")
        time.sleep(5) # 给 H5 动画一些缓冲时间

        # 2. 循环寻找并点击“热点”或“下一页”
        # 注意：你需要右键检查页面，找到那个热点按钮的特征（比如 class 包含 'hotspot' 或 'next-btn'）
        report_content = []
        
        # 假设报告有 10 页
        for i in range(1, 11):
            print(f"正在提取第 {i} 页内容...")
            
            # 获取当前页面所有可见文本
            # 很多 H5 会把文本放在 canvas 里，如果 get_attribute('innerText') 为空，
            # 则说明文字是图片，爬虫无法直接抓取，需结合 OCR。
            body_text = driver.find_element(By.TAG_NAME, "body").text
            report_content.append(f"--- 第 {i} 页 ---\n{body_text}")

            # 寻找热点按钮并点击
            try:
                # 这里根据你观察到的 'hotspot' 标签来定位
                # 示例：定位 class 为 'hotspot' 的元素
                next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "[class*='hotspot'], [class*='next']")))
                next_button.click()
                time.sleep(2) # 等待翻页动画
            except:
                print("未找到更多交互热点，可能已到末尾。")
                break

        # 保存结果
        with open("reading_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        print("抓取完成，结果已保存至 reading_report.txt")

    finally:
        driver.quit()

if __name__ == "__main__":
    extract_report()