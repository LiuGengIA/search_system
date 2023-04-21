// 获取搜索框的值
const searchTypeInput = document.getElementById('search-type');
const searchContentInput = document.getElementById('search-content');

// 在表单提交前保存搜索框的值
const assetSearchForm = document.getElementById('asset-search-form');
assetSearchForm.addEventListener('submit', function() {
    sessionStorage.setItem('search-type', searchTypeInput.value);
    sessionStorage.setItem('search-content', searchContentInput.value);
});

// 在页面加载时填充搜索框的值
const savedSearchType = sessionStorage.getItem('search-type');
const savedSearchContent = sessionStorage.getItem('search-content');
if (savedSearchType) {
    searchTypeInput.value = savedSearchType;
}
if (savedSearchContent) {
    searchContentInput.value = savedSearchContent;
}
