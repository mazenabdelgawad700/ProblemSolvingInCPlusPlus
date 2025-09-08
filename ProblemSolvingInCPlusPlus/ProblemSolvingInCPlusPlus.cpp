#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include<map>

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

int main()
{
	cout << isIsomorphic("badc", "baba") << endl;
    system("pause>0");
}