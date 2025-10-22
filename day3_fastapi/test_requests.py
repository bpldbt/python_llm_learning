import requests

def request_random_user():
    url = "https://randomuser.me/api/"
    print(f"正在请求 URL: {url}")

    try:
        reponse  = requests.get(url)

        if reponse.status_code == 200:
            data = reponse.json()
            user = data['results'][0]
            name = user['name']
            print("请求成功，返回的数据如下:")
            print(data)

        else:
                print(f"请求失败，状态码: {reponse.status_code}")
    except requests.RequestException as e:
        print(f"请求过程中出现错误: {e}")



if __name__ == "__main__":
    request_random_user()