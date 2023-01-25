# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None: # if head is None or head.next is None:
            return None # return head
        if head.next is None: # if head.next is None:
            return head # return head

        next = head.next
        head.next = self.swapPairs(next.next)
        next.next = head
        return next
