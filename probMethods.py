def normList(myList):
    alpha = 0
    for num in myList:
        alpha += num
    return [x / alpha for x in myList]
