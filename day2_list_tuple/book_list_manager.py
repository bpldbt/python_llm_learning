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

for fruit in fruits:
    print(fruit)

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
    
squares = [x**2 for x in range(1, 6)]
print(squares)

numbers = [10, 20, 30, 40, 50, 60, 70, 9]
even_numbers = [x for x in numbers if x % 2 == 0]
print(even_numbers)

my_book_list = []
my_book_list.append("Effective Python")
my_book_list.append("Clean Code")
my_book_list.append("The Pragmatic Programmer")
print("我的图书馆(初始):", my_book_list)
print("第一本书是:", my_book_list[0])
print("最后一本书是：", my_book_list[-1])
