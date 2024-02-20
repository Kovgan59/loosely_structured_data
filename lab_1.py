from zeep import Client 

def get_opt():
    print("Доступные категории для запроса:")
    print("+ Num Conversion")
    cat = input("Введите категорию запроса или оствьте поле не заполненым для выхода: ")

    if cat == "Num Conversion":
        url = "https://www.dataaccess.com/webservicesserver/NumberConversion.wso?WSDL"
    elif not(cat):
        return None
    else:
        result = str("Выбранная категория запроса не поддерживается.")
        return result
        
    if url:
        print("Доступные методы для выбранной категории:")
        client = Client(url)
        services = client.service
        methods = services.__dir__()
        filtered_methods = [method for method in methods if not method.startswith('__')]
        for method in filtered_methods:
            print(f"+ {method}")

        method = input("Введите метод: ")
        result = 0
        while result is 0:
            if method == "NumberToWords":
                ubiNum = input("Введите число: ")
                if not(ubiNum):
                    ubiNum = 0
                result = getattr(services, method)(ubiNum)
            elif method == "NumberToDollars":
                ubiNum = input("Введите число: ")
                if not(ubiNum):
                    ubiNum = 0
                result = getattr(services, method)(ubiNum)
            elif not(method):
                return None
            else:
                print("Выбран не существующий метод!")
                method = input("Введите метод: ")
        return result
    
while True: 
    result = get_opt() 
    if not(result): 
        print("Выход из запроса")
        break 
    print("Результат:", result)