import React, { useState, useEffect } from 'react';
import { Search, Edit2, Trash2, Plus, Save, X, Filter, UserPlus } from 'lucide-react';

const UserProfilesAdmin = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRole, setFilterRole] = useState('all');
  const [editingUser, setEditingUser] = useState(null);
  const [showAddUser, setShowAddUser] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8010';

  const roles = [
    'Property Owner',
    'Investor',
    'Real Estate Agent',
    'Developer',
    'Consultant'
  ];

  // Load users from Cosmos DB
  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      setLoading(true);
      setError('');
      
      const response = await fetch(`${API_BASE_URL}/api/users`);
      if (response.ok) {
        const userData = await response.json();
        setUsers(userData);
      } else {
        throw new Error('Failed to load users');
      }
    } catch (err) {
      console.error('Error loading users:', err);
      setError('Failed to load user profiles');
      
      // Fallback to demo data
      setUsers(getDemoUsers());
    } finally {
      setLoading(false);
    }
  };

  // Demo users for testing
  const getDemoUsers = () => [
    {
      id: 'demo-1',
      user_id: 'demo-1',
      email: 'sophie@investor.com',
      name: 'Sophie Investment',
      auth_type: 'google',
      user_role: 'Investor',
      suggested_questions: [
        'What ESG risks exist for EPC Class F?',
        'Which subsidies apply in Flanders?',
        'What renovations boost resale value?'
      ],
      created_at: '2025-01-15T10:00:00Z',
      last_login: '2025-01-20T14:30:00Z'
    },
    {
      id: 'demo-2',
      user_id: 'demo-2',
      email: 'marc@owner.be',
      name: 'Marc Proprietaire',
      auth_type: 'local',
      user_role: 'Property Owner',
      suggested_questions: [
        'How much will EPC Class E to B renovation cost?',
        'What grants are available in Brussels?',
        'When should I renovate to avoid 2030 penalties?'
      ],
      created_at: '2025-01-10T09:15:00Z',
      last_login: '2025-01-19T16:45:00Z'
    },
    {
      id: 'demo-3',
      user_id: 'demo-3',
      email: 'lisa@realtor.be',
      name: 'Lisa Agent',
      auth_type: 'google',
      user_role: 'Real Estate Agent',
      suggested_questions: [
        'How do I explain EPC impact to clients?',
        'What renovation advice should I give sellers?',
        'How do 2030 deadlines affect property marketing?'
      ],
      created_at: '2025-01-12T11:20:00Z',
      last_login: '2025-01-20T09:10:00Z'
    }
  ];

  // Filter users based on search and role
  const filteredUsers = users.filter(user => {
    const matchesSearch = !searchTerm || 
      user.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.name?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesRole = filterRole === 'all' || user.user_role === filterRole;
    
    return matchesSearch && matchesRole;
  });

  // Save user changes
  const saveUser = async (userData) => {
    try {
      const isNewUser = !userData.id;
      const url = isNewUser 
        ? `${API_BASE_URL}/api/users`
        : `${API_BASE_URL}/api/users/${userData.user_id}`;
      
      const method = isNewUser ? 'POST' : 'PUT';
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...userData,
          updated_at: new Date().toISOString()
        })
      });

      if (response.ok) {
        await loadUsers(); // Reload users list
        setEditingUser(null);
        setShowAddUser(false);
        return true;
      } else {
        throw new Error('Failed to save user');
      }
    } catch (err) {
      console.error('Error saving user:', err);
      setError('Failed to save user profile');
      return false;
    }
  };

  // Delete user
  const deleteUser = async (userId) => {
    if (!window.confirm('Are you sure you want to delete this user profile?')) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/users/${userId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        setUsers(users.filter(u => u.user_id !== userId));
      } else {
        throw new Error('Failed to delete user');
      }
    } catch (err) {
      console.error('Error deleting user:', err);
      setError('Failed to delete user profile');
    }
  };

  // Get default questions for role
  const getDefaultQuestionsForRole = (role) => {
    const questionSets = {
      'Investor': [
        'What ESG risks exist for EPC Class F properties?',
        'Which Belgian subsidies maximize ROI for renovations?',
        'What renovations boost resale value most?',
        'How do 2030 regulations affect rental property investments?'
      ],
      'Property Owner': [
        'How much will EPC Class E to B renovation cost?',
        'What grants are available in my region?',
        'When should I renovate to avoid 2030 penalties?',
        'What energy improvements add most value?'
      ],
      'Real Estate Agent': [
        'How do I explain EPC impact to clients?',
        'What renovation advice should I give sellers?',
        'How do 2030 deadlines affect property marketing?',
        'What ESG factors affect property valuations?'
      ],
      'Developer': [
        'What are the new construction ESG requirements?',
        'How do I optimize ESG scores for new projects?',
        'What sustainable materials qualify for incentives?',
        'How do ESG factors affect project financing?'
      ]
    };

    return questionSets[role] || questionSets['Property Owner'];
  };

  const UserEditModal = ({ user, onSave, onCancel }) => {
    const [formData, setFormData] = useState(user || {
      email: '',
      name: '',
      auth_type: 'local',
      user_role: 'Property Owner',
      suggested_questions: []
    });

    useEffect(() => {
      if (formData.user_role && (!formData.suggested_questions || formData.suggested_questions.length === 0)) {
        setFormData(prev => ({
          ...prev,
          suggested_questions: getDefaultQuestionsForRole(formData.user_role)
        }));
      }
    }, [formData.user_role]);

    const handleSubmit = (e) => {
      e.preventDefault();
      if (!formData.email || !formData.user_role) {
        alert('Email and role are required');
        return;
      }

      const userData = {
        ...formData,
        user_id: formData.user_id || `user_${Date.now()}`,
        created_at: formData.created_at || new Date().toISOString(),
        last_login: formData.last_login || new Date().toISOString()
      };

      onSave(userData);
    };

    const updateQuestion = (index, value) => {
      const newQuestions = [...formData.suggested_questions];
      newQuestions[index] = value;
      setFormData({...formData, suggested_questions: newQuestions});
    };

    const addQuestion = () => {
      setFormData({
        ...formData,
        suggested_questions: [...formData.suggested_questions, '']
      });
    };

    const removeQuestion = (index) => {
      const newQuestions = formData.suggested_questions.filter((_, i) => i !== index);
      setFormData({...formData, suggested_questions: newQuestions});
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          <div className="bg-blue-500 text-white p-4 flex justify-between items-center rounded-t-lg">
            <h3 className="text-lg font-semibold">
              {user ? 'Edit User Profile' : 'Add New User'}
            </h3>
            <button onClick={onCancel} className="hover:bg-blue-600 p-1 rounded">
              <X size={20} />
            </button>
          </div>
          
          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email *
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({...formData, name: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Role *
                </label>
                <select
                  value={formData.user_role}
                  onChange={(e) => setFormData({...formData, user_role: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                  required
                >
                  {roles.map(role => (
                    <option key={role} value={role}>{role}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Auth Type
                </label>
                <select
                  value={formData.auth_type}
                  onChange={(e) => setFormData({...formData, auth_type: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                >
                  <option value="local">Local</option>
                  <option value="google">Google</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Suggested Questions
              </label>
              <div className="space-y-2">
                {formData.suggested_questions?.map((question, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => updateQuestion(index, e.target.value)}
                      className="flex-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                      placeholder="Enter question..."
                    />
                    <button
                      type="button"
                      onClick={() => removeQuestion(index)}
                      className="px-3 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={addQuestion}
                  className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 flex items-center gap-2"
                >
                  <Plus size={16} />
                  Add Question
                </button>
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <button
                type="button"
                onClick={onCancel}
                className="flex-1 px-4 py-2 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
              >
                {user ? 'Update User' : 'Create User'}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2">Loading user profiles...</span>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-bold text-gray-800">User Profiles Management</h2>
        <button
          onClick={() => setShowAddUser(true)}
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 flex items-center gap-2"
        >
          <UserPlus size={16} />
          Add User
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-lg border p-4 mb-4">
        <div className="flex gap-4 items-center">
          <div className="flex-1">
            <div className="relative">
              <Search size={16} className="absolute left-3 top-3 text-gray-400" />
              <input
                type="text"
                placeholder="Search by email or name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <div>
            <select
              value={filterRole}
              onChange={(e) => setFilterRole(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Roles</option>
              {roles.map(role => (
                <option key={role} value={role}>{role}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-lg border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">User</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Auth Type</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Questions</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Last Login</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredUsers.map((user) => (
                <tr key={user.user_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <div>
                      <div className="font-medium text-gray-900">{user.name || 'N/A'}</div>
                      <div className="text-sm text-gray-500">{user.email}</div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 text-blue-800">
                      {user.user_role}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                      user.auth_type === 'google' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {user.auth_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {user.suggested_questions?.length || 0} questions
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setEditingUser(user)}
                        className="text-blue-600 hover:text-blue-800"
                        title="Edit user"
                      >
                        <Edit2 size={16} />
                      </button>
                      <button
                        onClick={() => deleteUser(user.user_id)}
                        className="text-red-600 hover:text-red-800"
                        title="Delete user"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredUsers.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500">No users found matching the current filters.</p>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="mt-4 grid grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">{users.length}</div>
          <div className="text-sm text-gray-600">Total Users</div>
        </div>
        <div className="bg-white rounded-lg border p-4 text-center">
          <div className="text-2xl font-bold text-green-600">
            {users.filter(u => u.user_role === 'Investor').length}
          </div>
          <div className="text-sm text-gray-600">Investors</div>
        </div>
        <div className="bg-white rounded-lg border p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">
            {users.filter(u => u.user_role === 'Property Owner').length}
          </div>
          <div className="text-sm text-gray-600">Property Owners</div>
        </div>
        <div className="bg-white rounded-lg border p-4 text-center">
          <div className="text-2xl font-bold text-orange-600">
            {users.filter(u => u.auth_type === 'google').length}
          </div>
          <div className="text-sm text-gray-600">Google Auth</div>
        </div>
      </div>

      {/* Modals */}
      {editingUser && (
        <UserEditModal
          user={editingUser}
          onSave={saveUser}
          onCancel={() => setEditingUser(null)}
        />
      )}

      {showAddUser && (
        <UserEditModal
          user={null}
          onSave={saveUser}
          onCancel={() => setShowAddUser(false)}
        />
      )}
    </div>
  );
};

export default UserProfilesAdmin;
