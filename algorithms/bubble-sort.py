# Bubble sort algorithm

def bubble_short(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

if __name__ == "__main__":
    arr = [1, 5, 2, 4, 3]
    print(bubble_short(arr))