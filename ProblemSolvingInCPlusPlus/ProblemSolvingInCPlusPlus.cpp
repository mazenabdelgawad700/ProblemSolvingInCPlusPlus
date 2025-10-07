#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include<map>
#include <set>
#include <unordered_set>

using namespace std;

bool isPalindrome(string s) {
    string sw = "";
    for(int i = 0; i < s.length(); i++) 
    {
        if(isalnum(s[i]))
        {
            sw += tolower(s[i]);
        }
	}
    int l = 0, r = sw.length() - 1;
    while (r > 0 && l != r)
    {
        if (sw[l] != sw[r])
            return false;
        
        l++;
        r--;
    }
    return true;
}

string convert(string s, int numRows) 
{
    if (numRows == 1)
        return s;
    string result = "";
    for (int r = 0; r < numRows; r++)
    {
        int increment = 2 * (numRows - 1);
        for (int i = r; i < s.length(); i += increment)
        {
            result += s[i];
            if (r > 0 && r < numRows - 1 && (i + increment - 2 * r) < s.length())
            {
                result += s[i + increment - 2 * r];
            }
        }
    }
    return result;
}

vector<int> twoSum(vector<int>& numbers, int target) {
    int l = 0, r = numbers.size() - 1;
    vector<int> res;
    while (l != r)
    {
        int sum = numbers.at(l) + numbers.at(r);
        if (sum > target)
            r--;
        else if (sum < target)
            l++;
        else
        {
            res.push_back(l + 1);
            res.push_back(r + 1);
            return res;
        }
    }
    return res;
}

int countStudents(vector<int>& students, vector<int>& sandwiches) {
	queue<int> q;
	for (int i = 0; i < students.size(); i++)
		q.push(students[i]);

	int count = 0;
	int index = 0;
	while (q.size() > 0 && count < sandwiches.size())
	{
		if (q.front() == sandwiches[index])
		{
			q.pop();
			index++;
			count = 0;
		}
		else
		{
			int temp = q.front();
			q.pop();
			q.push(temp);
			count++;
		}
	}

	return q.size();

}

bool isValid(string s) {
	stack<char> st;
	for (char c : s)
	{
		if (c == '(' || c == '{' || c == '[')
			st.push(c);
		else
		{
			if (st.empty())
				return false;
			char top = st.top();
			st.pop();
			if ((c == ')' && top != '(') || (c == '}' && top != '{') || (c == ']' && top != '['))
				return false;
		}
	}
	return st.empty();
}

struct ListNode {
	int val;
	ListNode* next;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}

};

ListNode* middleNode(ListNode* head)
{
	ListNode* slow = head;
	ListNode* fast = head;
	while (fast)
	{
		fast = fast->next;
		if (!fast)
			break;
		fast = fast->next;
		slow = slow->next;
	}
	return slow;
}

int maxArea(vector<int>& height) {
	int l = 0, r = height.size() - 1;
	int max_area = 0;
	while(l < r)
	{
		int area = min(height[l], height[r]) * (r - l);
		max_area = max(max_area, area);
		if (height[l] < height[r])
			l++;
		else
			r--;
	}
	return max_area;
}

vector<vector<int>> threeSum(vector<int>& nums) {
	int n = nums.size();
	sort(nums.begin(), nums.end());
	vector<vector<int>> res;
	for (int i = 0; i < n - 2; i++)
	{
		if (i > 0 && nums[i] == nums[i - 1])
			continue;
		int l = i + 1, r = n - 1;
		while (l < r)
		{
			int sum = nums[i] + nums[l] + nums[r];
			if (sum < 0)
				l++;
			else if (sum > 0)
				r--;
			else
			{
				res.push_back({ nums[i], nums[l], nums[r] });
				while (l < r && nums[l] == nums[l + 1])
					l++;
				while (l < r && nums[r] == nums[r - 1])
					r--;
				l++;
				r--;
			}
		}
	}
	return res;
}

int minSubArrayLen(int target, vector<int>& nums) {
	int n = nums.size();
	int l = 0, r = 0, sum = 0, min_len = INT_MAX;
	while (r < n)
	{
		sum += nums[r];
		while (sum >= target)
		{
			min_len = min(min_len, r - l + 1);
			sum -= nums[l];
			l++;
		}
		r++;
	}
	return min_len == INT_MAX ? 0 : min_len;
}

int lengthOfLongestSubstring(string s) {
	unordered_map<char, int> lastSeen; 
	int n = s.length();
	int l = 0, maxLen = 0;

	for (int r = 0; r < n; r++) 
	{
		char c = s[r];

		
		if (lastSeen.find(c) != lastSeen.end() && lastSeen[c] >= l) 
			l = lastSeen[c] + 1;

		lastSeen[c] = r; 

		maxLen = max(maxLen, r - l + 1);
	}

	return maxLen;
}

bool isValidSudoku(vector<vector<char>>& board) {

	for (int r = 0; r < 9; r++) { 
		bool seen[10] = { false };
		int xorSum = 0;
		for (int c = 0; c < 9; c++) { 
			char val = board[r][c];
			if (val != '.') {
				int num = val - '0';
				if (seen[num]) return false; 
				seen[num] = true;
				xorSum ^= num;
			}
		}
	}

	for (int c = 0; c < 9; c++) { 
		bool seen[10] = { false };
		int xorSum = 0;
		for (int r = 0; r < 9; r++) {
			char val = board[r][c];
			if (val != '.') {
				int num = val - '0';
				if (seen[num]) return false;
				seen[num] = true;
				xorSum ^= num;
			}
		}
	}

	for (int br = 0; br < 9; br += 3) {  
		for (int bc = 0; bc < 9; bc += 3) { 
			bool seen[10] = { false };
			int xorSum = 0;
			for (int r = 0; r < 3; r++) {      
				for (int c = 0; c < 3; c++) {
					char val = board[br + r][bc + c];
					if (val != '.') {
						int num = val - '0';
						if (seen[num]) return false;
						seen[num] = true;
						xorSum ^= num;
					}
				}
			}
		}
	}

	return true;
}

struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode() : val(0), left(nullptr), right(nullptr) {}
	TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
	TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}
	
};

TreeNode* build(vector<int>& nums, int left, int right) {
	if (left > right) return nullptr;

	int mid = left + (right - left) / 2; 
	TreeNode* root = new TreeNode(nums[mid]);

	root->left = build(nums, left, mid - 1);
	root->right = build(nums, mid + 1, right);

	return root;
}
TreeNode* sortedArrayToBST(vector<int>& nums) {
	return build(nums, 0, nums.size() - 1);
}

bool canConstruct(string ransomNote, string magazine) {
	map<char, int> dict;
	for (int i = 0; i < magazine.length(); i++)
	{
		if (dict.find(magazine[i]) == dict.end())
			dict[magazine[i]] = 1;
		else
			dict[magazine[i]]++;
	}

	for(int i = 0; i < ransomNote.length(); i++)
	{
		if (dict.find(ransomNote[i]) == dict.end())
			return false;
		else
		{
			if (!dict[ransomNote[i]])
				return false;
			else
				dict[ransomNote[i]]--;
		}
	}
	return true;
}

bool wordPattern(string pattern, string s) {
	map<char, string> charToWord;
	map<string, char> wordToChar;

	istringstream iss(s);
	vector<string> words;
	string word;

	while (iss >> word) 
	{
		words.push_back(word);
	}

	if (pattern.length() != words.size()) 
	{
		return false;
	}

	for (int i = 0; i < pattern.length(); i++) 
	{
		char c = pattern[i];
		string w = words[i];

		if (charToWord.count(c) && charToWord[c] != w) 
		{
			return false;
		}
		if (wordToChar.count(w) && wordToChar[w] != c) 
		{
			return false;
		}

		charToWord[c] = w;
		wordToChar[w] = c;
	}

	return true;
}

bool isIsomorphic(string s, string t) {
	map<char, char> dict;
	for (int i = 0; i < s.length(); i++)
	{
		if (dict.find(s[i]) == dict.end())
		{
			for (auto& c : dict)
			{
				if (c.second == t[i])
					return false;
			}
			dict[s[i]] = t[i];
		}
		else 
		{
			if (dict[s[i]] != t[i])
				return false;
		}
	}
	return true;
}
	
bool isAnagram(string s, string t) {
	map<char, int> dict;
	for (int i = 0; i < s.length(); i++)
	{
		if (dict.find(s[i]) == dict.end())
		{
			dict[s[i]] = 1;
		}
		else {
			dict[s[i]]++;
		}
	}

	for (int i = 0; i < t.length(); i++)
	{
		if (dict.find(t[i]) == dict.end() || !dict[t[i]])
			return false;
		else 
		{ 
			dict[t[i]]--;
		}
	}

	for (auto& c : dict)
	{
		if (c.second)
			return false;
	}

	return true;
}

bool isHappy(int n) {
	int temp = n, sum = 0;
	map<int, int> dict;
	while (true)
	{
		while (temp)
		{
			int digit = temp % 10;
			digit = pow(digit, 2);
			sum += digit;
			temp /= 10;
		}

		if (sum == 1)
			return true;

		if (dict.find(sum) != dict.end())
			return false;
		else
			dict[sum] = 1;

		temp = sum;
		sum = 0;
	}
	return false;
}

vector<int> twoSum2(vector<int>& nums, int target) {
	map<int, int> dict;
	vector<int> result;
	for (int i = 0; i < nums.size(); i++)
	{
		dict[nums[i]] = i;
	}

	for (int i = 0; i < nums.size(); i++)
	{
		int val = target - nums[i];
		if (dict.find(val) != dict.end() && i != dict[val])
		{
			result.push_back(i);
			result.push_back(dict[val]);
			return result;
		}
	}
	return result;
}

bool containsNearbyDuplicate(vector<int>& nums, int k) {
	map<int, int> dict;
	for (int i = 0; i < nums.size(); i++)
	{
		if (dict.find(nums[i]) != dict.end())
		{
			int res = abs(dict[nums[i]] - i);
			if (res <= k)
				return true;
			else
				dict[nums[i]] = i;
		}
		else
		{
			dict[nums[i]] = i;
		}
	}
	return false;
}

vector<vector<string>> groupAnagrams(vector<string>& strs) {
	map<string, vector<string>> dict;
	for (int i = 0; i < strs.size(); i++)
	{
		string s = strs[i];
		sort(s.begin(), s.end());
		dict[s].push_back(strs[i]);
	}
	vector<vector<string>> result;
	for (auto& entry : dict)
	{
		result.push_back(entry.second);
	}
	return result;
}

int longestConsecutive(vector<int>& nums) {
	unordered_set<int> numSet(nums.begin(), nums.end());
	int longest = 0;

	for (int num : numSet) {
		if (numSet.find(num - 1) == numSet.end()) {
			int currentNum = num;
			int length = 1;

			while (numSet.find(currentNum + 1) != numSet.end()) {
				currentNum++;
				length++;
			}

			longest = max(longest, length);
		}
	}
	return longest;
}

vector<string> summaryRanges(vector<int>& nums) {
	vector<string> result;
	int n = nums.size();
	for(int i = 0; i < n; i++)
	{
		int start = nums[i];
		while (i + 1 < n && nums[i + 1] == nums[i] + 1)
			i++;
		int end = nums[i];
		if (start == end)
			result.push_back(to_string(start));
		else
			result.push_back(to_string(start) + "->" + to_string(end));
	}
	return result;
}

string simplifyPath(string path) {
	vector<string> stack;
	string current = "";

	path += "/";

	for (char c : path) {
		if (c == '/') {
			if (current == "..") {
				if (!stack.empty()) {
					stack.pop_back();
				}
			}
			else if (current != "" && current != ".") {
				stack.push_back(current);
			}
			current = "";
		}
		else {
			current += c;
		}
	}

	string result = "/";
	for (const string& dir : stack) {
		result += dir + "/";
	}

	if (result.length() > 1) {
		result.pop_back();
	}

	return result;
}

class MinStack {
	stack<int> st;
	stack<int> minSt;
public:
	MinStack() {}

	void push(int val) {
		st.push(val);
		if (minSt.empty() || val <= minSt.top())
			minSt.push(val);
	}

	void pop() {
		if (st.top() == minSt.top())
			minSt.pop();
		st.pop();
	}

	int top() {
		return st.top();
	}

	int getMin() {
		return minSt.top();
	}
};

int evalRPN(vector<string>& tokens) {
	stack<int> st;
	for (int i = 0; i < tokens.size(); i++)
	{
		if (tokens.at(i) == "+" || tokens.at(i) == "*" || tokens.at(i) == "-" || tokens.at(i) == "/")
		{
			int b = st.top(); 
			st.pop();
			
			int a = st.top(); 
			st.pop();
			
			if (tokens[i] == "+") st.push(a + b);
			else if (tokens[i] == "-") st.push(a - b);
			else if (tokens[i] == "*") st.push(a * b);
			else if (tokens[i] == "/") st.push(a / b);
		}
		else
			st.push(stoi(tokens[i]));
	}
	return st.top();
}

bool hasCycle(ListNode* head) {
	set<ListNode*> set;
	ListNode* temp = head;
	while (temp)
	{
		if (set.find(temp) == set.end())
			set.insert(temp);
		else
			return true;
		temp = temp->next;
	}
	return false;
}

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
	ListNode* dummy = new ListNode(0);
	ListNode* current = dummy;
	while (list1 && list2)
	{
		if (list1->val < list2->val)
		{
			current->next = list1;
			list1 = list1->next;
		}
		else
		{
			current->next = list2;
			list2 = list2->next;
		}
		current = current->next;
	}
	if (list1)
		current->next = list1;
	if (list2)
		current->next = list2;
	return dummy->next;
}

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode dummy(0);
	ListNode* curr = &dummy;
	int carry = 0;

	while (l1 || l2 || carry) {
		int sum = carry;
		if (l1) {
			sum += l1->val;
			l1 = l1->next;
		}
		if (l2) {
			sum += l2->val;
			l2 = l2->next;
		}

		carry = sum / 10;
		curr->next = new ListNode(sum % 10);
		curr = curr->next;
	}

	return dummy.next;
}

class Node {
public:
	int val;
	Node* next;
	Node* random;

	Node(int _val) {
		val = _val;
		next = NULL;
		random = NULL;
	}
};

Node* copyRandomList(Node* head) {
	Node* newHead = new Node(0);
	map<Node*, Node*> dict;
	Node* temp = head;
	Node* curr = newHead;
	while (temp)
	{
		Node* newNode = new Node(temp->val);
		dict[temp] = newNode;
		curr->next = newNode;
		curr = curr->next;
		temp = temp->next;
	}
	temp = head;
	curr = newHead->next;
	while (temp)
	{
		if (temp->random)
			curr->random = dict[temp->random];
		temp = temp->next;
		curr = curr->next;
	}
	return newHead->next;
}

ListNode* reverseBetween(ListNode* head, int left, int right) {
	if (!head || left == right) return head;

	ListNode dummy(0);
	dummy.next = head;
	ListNode* prev = &dummy;

	for (int i = 0; i < left - 1; i++) {
		prev = prev->next;
	}

	ListNode* curr = prev->next;
	ListNode* next = nullptr;

	for (int i = 0; i < right - left + 1; i++) {
		next = curr->next;
		curr->next = prev->next;
		prev->next = curr;
		curr = next;
	}

	return dummy.next;
}

ListNode* deleteDuplicates(ListNode* head) {
	ListNode dummy(0);
	ListNode* curr = &dummy;
	dummy.next = head;

	while(curr->next && curr->next->next) {
		if (curr->next->val == curr->next->next->val) {
			int val = curr->next->val;
			while (curr->next && curr->next->val == val) {
				curr->next = curr->next->next;
			}
		} else {
			curr = curr->next;
		}
	}
	return dummy.next;
}

int maxDepth(TreeNode* root) {
	TreeNode* temp = root;
	if (!temp) return 0;
	int leftDepth = maxDepth(temp->left);
	int rightDepth = maxDepth(temp->right);
	return max(leftDepth, rightDepth) + 1;
}

bool isSameTree(TreeNode* p, TreeNode* q) {
	if (!p && !q)
		return true;
	if (!p || !q)
		return false;
	if (p->val != q->val)
		return false;
	return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

TreeNode* invertTree(TreeNode* root) {
	if (root->right)
	{
		invertTree(root->right);
	}
	if (root->left)
	{
		invertTree(root->left);
	}
	TreeNode* temp = root->left;
	root->left = root->right;
	root->right = temp;
	return root;
}

bool isMirror(TreeNode* t1, TreeNode* t2) {
	if (!t1 || !t2) return false;
	if (!t1 && !t2) return true;
	if (t1->val != t2->val) return false;
	return isMirror(t1->left, t2->right) && isMirror(t1->right, t2->left);
}

bool isSymmetric(TreeNode* root) {
	if (!root) return false;
	return isMirror(root->left, root->right);
}

bool hasPathSum(TreeNode* root, int targetSum) {
	if (!root) return false;
	if(!root->left && !root->right && targetSum - root->val == 0)
		return true;
	return hasPathSum(root->left, targetSum - root->val) || hasPathSum(root->right, targetSum - root->val);
}

void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	int i = m - 1, j = n - 1, k = m + n - 1;
 	while (i >= 0 && j >= 0) { 
		if (nums1[i] > nums2[j]) {
			nums1[k--] = nums1[i--];
		} else {
			nums1[k--] = nums2[j--];
		}
	}
	while (j >= 0) {
		nums1[k--] = nums2[j--];
	}
}

int thirdMax(vector<int>& nums) {
	vector<int> uniqueNums;
	for (int num : nums) {
		if (find(uniqueNums.begin(), uniqueNums.end(), num) == uniqueNums.end()) {
			uniqueNums.push_back(num);
		}
	}
	sort(uniqueNums.begin(), uniqueNums.end(), greater<int>());
	if (uniqueNums.size() < 3) {
		return uniqueNums[0];
	} else {
		return uniqueNums[2];
	}
}

vector<int> smallerNumbersThanCurrent(vector<int>& nums) {
	vector<int> result;
	for(int i = 0; i < nums.size(); i++)
	{
		int count = 0;
		for (int j = 0; j < nums.size(); j++)
		{
			if (nums[j] < nums[i])
				count++;
		}
		result.push_back(count);
	}
	return result;
}

void flatten(TreeNode* root) {
	TreeNode* temp = root;
	while(temp)
	{
		if (temp->left)
		{
			TreeNode* rightmost = temp->left;
			while (rightmost->right)
			{
				rightmost = rightmost->right;
			}
			rightmost->right = temp->right;
			temp->right = temp->left;
			temp->left = nullptr;
		}
		temp = temp->right;
	}
}

int sumNumbersHelper(TreeNode* root, int currentSum)
{
	if (!root) return 0;
	currentSum = currentSum * 10 + root->val;
	if (!root->left && !root->right)
		return currentSum;
	return sumNumbersHelper(root->left, currentSum) + sumNumbersHelper(root->right, currentSum);
}
int sumNumbers(TreeNode* root) {
	TreeNode* temp = root;
	if (!temp) return 0;
	int sum = 0;
	sum = sumNumbersHelper(temp, 0);
	return sum;
}

vector<int> rightSideView(TreeNode* root)
{
	TreeNode* temp = root;
	vector<int> result;
	if (!temp) return result;
	queue<TreeNode*> q;

	q.push(temp);
	
	while (!q.empty())
	{
		int size = q.size();
		for (int i = 0; i < size; i++)
		{
			TreeNode* node = q.front();
			q.pop();
			if (i == size - 1)
				result.push_back(node->val);
			if (node->left)
				q.push(node->left);
			if (node->right)
				q.push(node->right);
		}
	}
	return result;
}

int maxFrequencyElements(vector<int>& nums) {
	int* arr = new int[101](); 

	for (int i = 0; i < nums.size(); i++) {
		arr[nums[i]]++;
	}

	int maxFreq = 0;
	for (int i = 0; i < 101; i++) {
		if (arr[i] > maxFreq) {
			maxFreq = arr[i];
		}
	}

	int count = 0;
	for (int i = 0; i < 101; i++) {
		if (arr[i] == maxFreq) {
			count += arr[i]; 
		}
	}

	delete[] arr;
	return count;
}

int searchInsert(vector<int>& nums, int target) {
	int l = 0, r = nums.size() - 1, med = 0;
	
	while (r >= l)
	{
		med = (l + r) / 2;
		if (nums[med] == target)
			return med;
		else if (target < nums[med])
			r = med - 1;
		else
			l = med + 1;

	}
	return med + 1;
}

vector<int> plusOne(vector<int>& digits) {
	int last_digit = digits[digits.size() - 1];
	if (last_digit < 9)
	{
		digits[digits.size() - 1] += 1;
		return digits;
	}
	else
	{
		int index = digits.size() - 1;
		while (digits[index] == 9 && index > 0)
		{
			digits[index] = 0;
			index--;
		}
		if (digits[index] == 9)
		{
			digits[index] = 0;
			digits.insert(digits.begin(), 1);
		}
		else 
			digits[index] += 1;
		
	}
	return digits;

}

int mySqrt(int x) {
	if (x == 0) return 0;
	int left = 1, right = x, ans = 0;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (mid <= x / mid) {
			ans = mid; 
			left = mid + 1; 
		} else {
			right = mid - 1; 
		}
	}
	return ans;
}

int climbStairs(int n)
{
	if (n <= 2) return n;
	int first = 1, second = 2, third = 0;
	for (int i = 3; i <= n; i++)
	{
		third = first + second;
		first = second;
		second = third;
	}
	return third;
}

ListNode* deleteDuplicates2(ListNode* head) {
	ListNode* current = head;
	while (current && current->next)
	{
		if(current->val == current->next->val)
		{
			ListNode* temp = current->next;
			current->next = current->next->next;
			delete temp; 
		}
		else
		{
			current = current->next;
		}
	}
	return head;
}

vector<int> inorderTraversal(TreeNode* root) {
	if (!root) return {};
	vector<int> result;
	stack<TreeNode*> st;
	TreeNode* current = root;
	while (current || !st.empty())
	{
		while(current)
		{
			st.push(current);
			current = current->left;
		}
		if (!st.empty())
		{
			current = st.top();
			st.pop();
			result.push_back(current->val);
			current = current->right;
		}
	}
	return result;
}

int calcHeight(TreeNode* root)
{
	if (!root) return 0;
	int left_height = calcHeight(root->left);
	int right_height = calcHeight(root->right);
	return max(left_height, right_height) + 1;
}
bool isBalanced(TreeNode* root) {
	if (!root) return true;
	int left_height = calcHeight(root->left);
	int right_height = calcHeight(root->right);
	if (abs(left_height - right_height) > 1)
		return false;
	return isBalanced(root->left) && isBalanced(root->right);
}

ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
	set<ListNode*> nodesInA;
	ListNode* currentA = headA;
	while (currentA) {
		nodesInA.insert(currentA);
		currentA = currentA->next;
	}
	ListNode* currentB = headB;
	while (currentB) {
		if (nodesInA.find(currentB) != nodesInA.end()) {
			return currentB; 
		}
		currentB = currentB->next;
	}
	return nullptr;
}

ListNode* removeElements(ListNode* head, int val)
{
	ListNode* dummy = new ListNode(0);
	dummy->next = head;
	ListNode* current = dummy;
	while (current->next)
	{
		if (current->next->val == val)
		{
			current->next = current->next->next;
		}
		else
		{
			current = current->next;
		}
	}
	return dummy->next;
}

string convertToTitle(int columnNumber) {
	string result = "";
	map<int, char> alphaMap = {
	{0, 'A'}, {1, 'B'}, {2, 'C'}, {3, 'D'}, {4, 'E'}, {5, 'F'},
	{6, 'G'}, {7, 'H'}, {8, 'I'}, {9, 'J'}, {10, 'K'}, {11, 'L'},
	{12, 'M'}, {13, 'N'}, {14, 'O'}, {15, 'P'}, {16, 'Q'}, {17, 'R'},
	{18, 'S'}, {19, 'T'}, {20, 'U'}, {21, 'V'}, {22, 'W'}, {23, 'X'},
	{24, 'Y'}, {25, 'Z'}
	};

	while (columnNumber)
	{
		result.insert(result.begin(), alphaMap[(columnNumber - 1) % 26]);
		columnNumber -= 1;
		columnNumber /= 26;
	}
	return result;
}

ListNode* reverseList(ListNode* head) {
	ListNode* prev = nullptr;
	ListNode* current = head;
	while (current)
	{
		ListNode* nextTemp = current->next;
		current->next = prev;
		prev = current;
		current = nextTemp;
	}
	return prev;
}

bool isPowerOfTwo(int n) {
	int l = 0, r = n - 1;
	while (l <= r)
	{
		int mid = (l + r) / 2;
		if (pow(2, mid) == n)
			return true;
		else if (pow(2, mid) > n)
		{
			r = mid - 1;
		}
		else
		{
			l = mid + 1;
		}
	}
	return false;
}

vector<string> binaryTreePaths(TreeNode* root) {
	vector<string> result;
	if (!root) return result;
	stack<pair<TreeNode*, string>> st;
	st.push({ root, to_string(root->val) });
	while (!st.empty())
	{
		auto& top = st.top();
		TreeNode* node = top.first;
		string path = top.second;

		st.pop();

		if (!node->left && !node->right)
		{
			result.push_back(path);
		}
		if (node->right)
			st.push({ node->right, path + "->" + to_string(node->right->val) });
		if (node->left)
			st.push({ node->left, path + "->" + to_string(node->left->val) });

	}
	return result;
}

void moveZeroes(vector<int>& nums) {
	for (int i = 0; i < nums.size(); i++)
	{
		for (int j = i + 1; j < nums.size(); j++)
		{
			if (!nums[i])
			{
				nums[i] = nums[j];
				nums[j] = 0;
			}
		}
	}
}

vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
	map<int, int> dict;
	vector<int> result;
	
	for (int i = 0; i < nums1.size(); i++)
	{
		if (dict.find(nums1[i]) == dict.end())
			dict[nums1[i]] = 1;
		else
			dict[nums1[i]]++;
	}

	for (int i = 0; i < nums2.size(); i++)
	{
		if (dict.find(nums2[i]) != dict.end() && dict[nums2[i]])
		{
			result.push_back(nums2[i]);
			dict[nums2[i]]--;
		}
	}
	return result;
}

bool isPerfectSquare(int num) {
	if (num < 2) return true;
	int left = 2, right = num / 2;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		long long guessedSquare = (long long)mid * mid;
		if (guessedSquare == num) {
			return true;
		} else if (guessedSquare < num) {
			left = mid + 1;
		} else {
			right = mid - 1;
		}
	}
	return false;
}

int firstUniqChar(string s) {
	unordered_map<char, int> dict;
	for (int i = 0; i < s.length(); i++)
	{
		if (dict.find(s[i]) != dict.end())
			dict[s[i]] = -1;
		else 
			dict[s[i]] = 1;
	}

	char character = ' ';
	for (const auto& p : dict)
	{
		if (p.second == 1)
		{
			character = p.first;
			break;
		}
	}
	for (int i = 0; i < s.length(); i++)
		if (s[i] == character)
			return i;

	return -1;
}

int sumOfLeftLeaves(TreeNode* root) {
	queue<TreeNode*> q;
	int sum = 0;
	q.push(root);
	while (!q.empty())
	{
		TreeNode* node = q.front();
		q.pop();
		if (node->left)
		{
			if (!node->left->left && !node->left->right)
				sum += node->left->val;
			else
				q.push(node->left);
		}
		if(node->right)
			q.push(node->right);

	}
	return sum;
}

int longestPalindrome(string s) {
	map<char, int> dict;
	for (int i = 0; i < s.length(); i++)
	{
		if (dict.find(s[i]) != dict.end())
			dict[s[i]]++;
		else
			dict[s[i]] = 1;
	}

	int length = 0;
	bool odd_flag = false;
	for (const auto& p : dict)
	{
		if (p.second % 2 == 0)
			length += p.second;
		else if (p.second % 2 != 0)
		{
			length += p.second - 1;
			odd_flag = true;
		}
	}
	if (odd_flag)
		length += 1;

	return length;
}

vector<string> fizzBuzz(int n) {
	vector<string> result = {};
	for (int i = 1; i <= n; i++)
	{
		if (i % 3 == 0 && i % 5 == 0)
			result.push_back("FizzBuzz");
		else if (i % 3 == 0)
			result.push_back("Fizz");
		else if (i % 5 == 0)
			result.push_back("Buzz");
		else
			result.push_back(to_string(i));
	}
	return result;
}

vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> result;
	if (!root) return result;
	queue<TreeNode*> q;
	q.push(root);
	while(!q.empty())
	{
		int size = q.size();
		vector<int> level;
		for (int i = 0; i < size; i++)
		{
			TreeNode* node = q.front();
			q.pop();
			level.push_back(node->val);
			if (node->left)
				q.push(node->left);
			if (node->right)
				q.push(node->right);
		}
		result.push_back(level);
	}
	return result;
}

int arrangeCoins(int n) {
	int k = 0;
	while (n >= 0)
	{
		k++;
		n -= k;
	}
	return k - 1;
}

bool repeatedSubstringPattern(string s) {
	int n = s.length();
	for (int i = 1; i <= n / 2; i++)
	{
		if (n % i == 0)
		{
			string substring = s.substr(0, i);
			string repeated = "";
			for (int j = 0; j < n / i; j++)
				repeated += substring;
			if (repeated == s)
				return true;
		}
	}
	return false;
}

int fib(int n) {
	if (n <= 2) return 1;
	return fib(n - 1) + fib(n - 2);
}

vector<int> constructRectangle(int area) {
	vector<int> result(2);
	int width = sqrt(area);
	while (area % width != 0) {
		width--;
	}
	result[0] = area / width;
	result[1] = width;
	return result;
}

vector<int> findMode(TreeNode* root) {
	map<int, int> countMap;
	int max_count = 0;
	TreeNode* current = root;
	queue<TreeNode*> q;
	q.push(current);

	while(!q.empty())
	{
		TreeNode* node = q.front();
		q.pop();
		countMap[node->val]++;
		max_count = max(max_count, countMap[node->val]);
		if (node->left)
			q.push(node->left);
		if (node->right)
			q.push(node->right);
	}
	vector<int> modes;
	for (const auto& p : countMap)
	{
		if (p.second == max_count)
			modes.push_back(p.first);
	}
	return modes;
}

vector<string> findRelativeRanks(vector<int>& score) {
	vector<string> result(score.size());
	vector<int> sortedScore = score;
	sort(sortedScore.begin(), sortedScore.end(), greater<int>());
	map<int, string> rankMap;
	for (int i = 0; i < sortedScore.size(); i++)
	{
		if (i == 0)
			rankMap[sortedScore[i]] = "Gold Medal";
		else if (i == 1)
			rankMap[sortedScore[i]] = "Silver Medal";
		else if (i == 2)
			rankMap[sortedScore[i]] = "Bronze Medal";
		else
			rankMap[sortedScore[i]] = to_string(i + 1);
	}
	for (int i = 0; i < score.size(); i++)
	{
		result[i] = rankMap[score[i]];
	}
	return result;
}

bool detectCapitalUse(string word)
{
	if (word.length() == 1)
		return true;

	bool isFirstUpper = isupper(word[0]);
	bool isSecondUpper = isupper(word[1]);

	if (!isFirstUpper && isSecondUpper) {
		return false;
	}

	for (int i = 2; i < word.length(); i++) {
		if (isFirstUpper && isSecondUpper) {
			if (!isupper(word[i]))
				return false;
		}
		else {
			if (isupper(word[i]))
				return false;
		}
	}
	return true;
}

map<int, int> storage;
int tribonacci(int n) {
	if (n == 0)
		return 0;
	if (n == 1 || n == 2)
		return 1;
	if (storage.find(n) != storage.end())
		return storage[n];

	storage[n] = tribonacci(n - 1) + tribonacci(n - 2) + tribonacci(n - 3);

	return storage[n];
}

void num_islands_dfs(vector<vector<char>>& grid, int i, int j)
{
	if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size())
		return;

	if (grid[i][j] == '0')
		return;

	grid[i][j] = '0';
	num_islands_dfs(grid, i, j + 1); // right
	num_islands_dfs(grid, i + 1, j); // bottom
	num_islands_dfs(grid, i, j - 1); // left
	num_islands_dfs(grid, i - 1, j); // top
}
int numIslands(vector<vector<char>>& grid) {
	int rows = grid.size();
	int cols = grid[0].size();
	int islands = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (grid[i][j] == '1')
			{
				islands++;
				num_islands_dfs(grid, i, j);
			}
		}
	}
	return islands;
}

vector<vector<int>> mergeIntervals(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end());

	vector<vector<int>> output;
	output.push_back(intervals[0]);

	for (int i = 1; i < intervals.size(); ++i) {
		int lastEnd = output.back()[1];
		int start = intervals[i][0];
		int end = intervals[i][1];

		if (start <= lastEnd) {
			output.back()[1] = max(lastEnd, end);
		}
		else {
			output.push_back({ start, end });
		}
	}

	return output;
}

bool compareTree(TreeNode* root, TreeNode* subRoot)
{
	if (!root && !subRoot)
		return true;

	if (!root || !subRoot)
		return false;

	if (root->val != subRoot->val)
		return false;

	return compareTree(root->left, subRoot->left) && compareTree(root->right, subRoot->right);
}
bool isSubtree(TreeNode* root, TreeNode* subRoot)
{
	if (!root)
		return false;

	if (compareTree(root, subRoot))
		return true;

	return isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot);
}

class MedianFinder {
public:
	priority_queue<int> low;
	priority_queue<int, vector<int>, greater<int>> high;
	MedianFinder() {}

	void addNum(int num) 
	{
		low.push(num);

		high.push(low.top());
		low.pop();

		if (low.size() < high.size())
		{
			low.push(high.top());
			high.pop();
		}
	}

	double findMedian() {
		if (low.size() > high.size())
			return low.top();

		return (low.top() + high.top()) / 2.0;
	}
};

int main()
{ 
	system("pause>0");
}