#include <stdio.h>
#include <stdlib.h>

typedef struct dnode {
    int value;
    struct dnode* next;
    struct dnode* last;
} DNode;

// 初始化双链表
DNode* init_double_linked_list(int arr[], int length) {
    if (length <= 0) return NULL;
    
    // 创建头结点
    DNode* head = (DNode*)malloc(sizeof(DNode));
    head->value = arr[0];
    head->last = NULL;
    head->next = NULL;
    DNode* last = head;
    
    // 创建后续结点
    for (int i = 1; i < length; i++) {
        DNode* newNode = (DNode*)malloc(sizeof(DNode));
        newNode->value = arr[i];
        newNode->last = last;
        newNode->next = NULL;
        
        last->next = newNode;
        last = newNode;
    }
    return head;
}

void show_double_linked_list(DNode* node, int direction) {
    DNode* current = node;
    if (direction) {
        // 正向打印
        while (current) {
            printf("%d ", current->value);
            current = current->next;
        }
    } else {
        // 反向打印
        while (current) {
            printf("%d ", current->value);
            current = current->last;
        }
    }
    printf("\n");
}

int main() {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    DNode* head = init_double_linked_list(arr, 10);
    // 正向
    show_double_linked_list(head, 1);
    // 反向
    DNode* tail = head;
    while (tail->next) {
        tail = tail->next;
    }
    show_double_linked_list(tail, 0);
    
    return 0;
}
