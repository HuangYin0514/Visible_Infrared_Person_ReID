if __name__ == "__main__":

    for i in range(10):
        for j in range(10):
            if j < 5:
                continue
            print(j, end=" ")
        print()
