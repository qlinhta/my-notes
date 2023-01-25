# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 1:
            return head
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        while prev:
            prev = self.reverse(prev, k)
        return dummy.next

    def reverse(self, prev, k):
        last = prev
        for _ in range(k + 1):
            last = last.next
            if not last and _ < k:
                return None
        tail = prev.next
        curr = prev.next.next
        while curr != last:
            next = curr.next
            curr.next = prev.next
            prev.next = curr
            curr = next
        tail.next = last
        return tail
