class Logger:

    def __init__(self, file_path):
        self.file_path = file_path

    def __call__(self, input):
        input = str(input)
        with open(self.file_path, "a") as f:
            f.writelines(input + "\n")
        print(input)
