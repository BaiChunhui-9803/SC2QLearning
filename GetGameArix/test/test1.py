students = []
count1 = 0
count2 =0
average =0.0
while True:
    info = input("请输入学生姓名、性别、年龄（用空格隔开）：")
    if not info:
        break
    info_list = info.split()
    students.append(info_list)
print("学生信息如下：")
for student in students:
    print("姓名：{}，性别：{}，年龄：{}".format(student[0], student[1], student[2]))
    average += float(student[2])
    count2 += 1
    if (student[1]=='男'):
        count1 += 1
print("平均年龄是{}，男性人数是{}".format((average/count2),count1))