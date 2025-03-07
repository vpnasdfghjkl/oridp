// Auto-generated. Do not edit!

// (in-package dynamic_biped.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class controlEndHandRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.left_hand_position = null;
      this.right_hand_position = null;
    }
    else {
      if (initObj.hasOwnProperty('left_hand_position')) {
        this.left_hand_position = initObj.left_hand_position
      }
      else {
        this.left_hand_position = [];
      }
      if (initObj.hasOwnProperty('right_hand_position')) {
        this.right_hand_position = initObj.right_hand_position
      }
      else {
        this.right_hand_position = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type controlEndHandRequest
    // Serialize message field [left_hand_position]
    bufferOffset = _arraySerializer.uint8(obj.left_hand_position, buffer, bufferOffset, null);
    // Serialize message field [right_hand_position]
    bufferOffset = _arraySerializer.uint8(obj.right_hand_position, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type controlEndHandRequest
    let len;
    let data = new controlEndHandRequest(null);
    // Deserialize message field [left_hand_position]
    data.left_hand_position = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    // Deserialize message field [right_hand_position]
    data.right_hand_position = _arrayDeserializer.uint8(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.left_hand_position.length;
    length += object.right_hand_position.length;
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/controlEndHandRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1790ef3209367c45321962dc0cfca107';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8[] left_hand_position
    uint8[] right_hand_position
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new controlEndHandRequest(null);
    if (msg.left_hand_position !== undefined) {
      resolved.left_hand_position = msg.left_hand_position;
    }
    else {
      resolved.left_hand_position = []
    }

    if (msg.right_hand_position !== undefined) {
      resolved.right_hand_position = msg.right_hand_position;
    }
    else {
      resolved.right_hand_position = []
    }

    return resolved;
    }
};

class controlEndHandResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.result = null;
    }
    else {
      if (initObj.hasOwnProperty('result')) {
        this.result = initObj.result
      }
      else {
        this.result = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type controlEndHandResponse
    // Serialize message field [result]
    bufferOffset = _serializer.bool(obj.result, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type controlEndHandResponse
    let len;
    let data = new controlEndHandResponse(null);
    // Deserialize message field [result]
    data.result = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'dynamic_biped/controlEndHandResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eb13ac1f1354ccecb7941ee8fa2192e8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool result
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new controlEndHandResponse(null);
    if (msg.result !== undefined) {
      resolved.result = msg.result;
    }
    else {
      resolved.result = false
    }

    return resolved;
    }
};

module.exports = {
  Request: controlEndHandRequest,
  Response: controlEndHandResponse,
  md5sum() { return '741989817a59889e258aa9e94c7ada8a'; },
  datatype() { return 'dynamic_biped/controlEndHand'; }
};
