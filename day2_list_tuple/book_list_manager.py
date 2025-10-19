empty_list = []
fruits = ["apple", "banana", "cherry"]
mixed_list = ["hello", 123, True]

fruits[1] = 123
fruits.append("123")

more_fruits = ["mango", "pear"]
fruits.extend(more_fruits)

print(fruits)

del fruits[0]
print(fruits)
pop_result = fruits.pop()
print(pop_result)
fruits.remove("mango")
print(fruits)
print(len(fruits))

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
    
