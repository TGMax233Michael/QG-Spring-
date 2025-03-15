#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int value;
    struct Node* next;
}Node;

// 初始化链表
int init_linked_list(int arr[], int length) {
    if (length == 0) return;

    // 创建头结点
    Node* head = (Node*)malloc(sizeof(Node));
    head->value = arr[0];
    head->next = NULL;
    Node* last = head;
    
    // 创建后续节点
    for (int i=1; i<length; i++){
        Node* newNode = (Node*)malloc(sizeof(Node));
        newNode->value = arr[i];
        newNode->next = NULL;

        last->next = newNode;
        last = last->next;
    }

    return head;
}

void show_linked_list(Node* head){
    Node* current = head;

    while (current){
        printf("%d ", current->value);
        current = current->next;
    }
}

int main(){
    Node* head = NULL;
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    head = init_linked_list(arr, 10);
    show_linked_list(head);

    return 0;
}


