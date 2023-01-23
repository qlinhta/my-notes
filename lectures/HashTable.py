class HashTable:
    def __init__(self):
        self.size = 10
        self.keys = [None] * self.size
        self.values = [None] * self.size

    def put(self, key, value):
        index = self.hash(key)
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return
            index = (index + 1) % self.size
        self.keys[index] = key
        self.values[index] = value

    def get(self, key):
        index = self.hash(key)
        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.size
        return None

    def hash(self, key):
        hash = 0
        for i in range(len(key)):
            hash = (hash * 31 + ord(key[i])) % self.size
        return hash

    def __setitem__(self, key, value):
        self.put(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def __len__(self):
        return self.size

    def __contains__(self, key):
        return key in self.keys

    def __delitem__(self, key):
        index = self.hash(key)
        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.keys[index] = None
                self.values[index] = None
            index = (index + 1) % self.size

    def __str__(self):
        return str(self.keys) + " " + str(self.values)

    def __repr__(self):
        return str(self.keys) + " " + str(self.values)

    def __iter__(self):
        return iter(self.keys)


def main():
    table = HashTable()
    table["one"] = 1
    table["two"] = 2
    table["three"] = 3
    table["four"] = 4

    print(table)

    print(table["one"])
    print(table["two"])
    print(table["three"])
    print(table["four"])

    print("one" in table)
    print("two" in table)
    print("three" in table)
    print("four" in table)


if __name__ == "__main__":
    main()
