# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: list[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        left = self.mergeKLists(lists[:mid])
        right = self.mergeKLists(lists[mid:])
        return self.mergeTwoLists(left, right)

    def mergeTwoLists(self, left, right):
        if left is None:
            return right
        if right is None:
            return left

        if left.val < right.val:
            left.next = self.mergeTwoLists(left.next, right)
            return left
        else:
            right.next = self.mergeTwoLists(left, right.next)
            return right
