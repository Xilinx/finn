/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the state machine implementation
 * for the SMBus driver.
 *
 * @file smbus_state.c
 *
 */

#include "smbus.h"
#include "smbus_internal.h"
#include "smbus_state.h"
#include "smbus_event.h"
#include "smbus_action.h"
#include "smbus_hardware_access.h"
#include "smbus_hardware.h"

char* pcStateSMBusStateInitial                  = "SMBUS_STATE_INITIAL";
char* pcStateSMBusStateAwaitingCommandByte      = "SMBUS_STATE_AWAITING_COMMAND_BYTE";
char* pcStateSMBusStateAwaitingBlockSize        = "SMBUS_STATE_AWAITING_BLOCK_SIZE";
char* pcStateSMBusStateAwaitingData             = "SMBUS_STATE_AWAITING_DATA";
char* pcStateSMBusStateAwaitingRead             = "SMBUS_STATE_AWAITING_READ";
char* pcStateSMBusStateReadyToSendByte          = "SMBUS_STATE_READY_TO_SEND_BYTE";
char* pcStateSMBusStateCheckIfPecRequired       = "SMBUS_STATE_CHECK_IF_PEC_REQUIRED";
char* pcStateSMBusStateAwaitingDone             = "SMBUS_STATE_AWAITING_DONE";
char* pcStateSMBusStateControllerSendCommand    = "SMBUS_STATE_CONTROLLER_SEND_COMMAND";
char* pcStateSMBusStateControllerSendReadStart  = "SMBUS_STATE_CONTROLLER_SEND_READ_START";
char* pcStateSMBusStateControllerReadBlockSize  = "SMBUS_STATE_CONTROLLER_READ_BLOCK_SIZE";
char* pcStateSMBusStateControllerReadByte       = "SMBUS_STATE_CONTROLLER_READ_BYTE";
char* pcStateSMBusStateControllerReadPec        = "SMBUS_STATE_CONTROLLER_READ_PEC";
char* pcStateSMBusStateControllerReadDone       = "SMBUS_STATE_CONTROLLER_READ_DONE";
char* pcStateSMBusStateControllerWriteByte      = "SMBUS_STATE_CONTROLLER_WRITE_BYTE";
char* pStateUnknown                             = "SMBUS_STATE_UNKNOWN";

/********************** Static function declarations ***************************/

/******************************************************************************
*
* @brief    Log an error when an event the FSM does not expect arrives
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is the latest event being handled
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vSMBusLogUnexpected( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Move current state to previous state and set the newstate as current
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    xNewState is the latest FSM state to be recorded
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vSMBusNextStateDecoder( SMBUS_INSTANCE_TYPE* pxSMBusInstance, SMBus_State_Type xNewState );

/******************************************************************************
*
* @brief    Perform actions on receiving an Prepare To ARP command
*           Clear AR flag
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vSMBusARPPrepareToARP( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Perform actions on receiving an ARP reset device command
*           AV and AR flags will be set dependent on the ARP capabilities
*           of the instance
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vSMBusARPResetDevice( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Write a descriptor to the IP to send a NACK
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vSMBusSendNACK( SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    This function swaps the locations of two SMBus instance numbers
*           within an array of SMBus instances
*
* @param    pucInstanceA is an SMBus instance number
* @param    pucInstanceB is an SMBus instance number
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vInstanceSwap( uint8_t* pucInstanceA, uint8_t* pucInstanceB );

/******************************************************************************
*
* @brief    This function takes an array of SMBus instances
*           It walks through that array and reorders the instances
*           in the order that they should respond to a ARP GetUDID general command
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure
* @param    pucInstanceIDArray is the array of SMBus instance numbers
* @param    ucArraySize is nmber of SMBus instance numbers in the array
* @param    ucUDIDByteIndex is the number of UDID bytes needed to determine the order
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vUDIDInstanceSort( SMBUS_PROFILE_TYPE* pxTheProfile, uint8_t pucInstanceIDArray[], uint8_t ucArraySize , uint8_t ucUDIDByteIndex );

/******************************************************************************
*
* @brief    This function walks through each instance to determine if
*           the UDID transmitted matches any one of the active SMBus instances
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
*
* @return   SMBUS_FALSE If no match is found
*           SMBUS_TRUE  If a match is found
*
* @note     None.
*
*****************************************************************************/
static uint8_t ucCheckAtLeastOneMatchFound( SMBUS_PROFILE_TYPE* pxSMBusProfile, SMBUS_INSTANCE_TYPE* pxSMBusInstance );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is Initial
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateInitial( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingCommandByte
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingCommandByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingRead
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingRead( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ReadyToSendByte
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateReadyToSendByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is CheckIfPECRequired
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateCheckIfPECRequired( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingDone
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingDone( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingData
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingData( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingBlockSize
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingBlockSize( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerSendCommand
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerSendCommand( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerWriteByte
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerWriteByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerSendReadStart
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerSendReadStart( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadBlockSize
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadBlockSize( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadByte
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadPEC
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadPEC( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadDone
*
* @param    pxSMBusInstance is a pointer to the SMBus instance structure.
* @param    ucAnyEvent is an event triggered by the driver or by an interrupt
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadDone( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent );

/*******************************************************************************/

/* Descriptor helper functions */

/******************************************************************************
*
* @brief    Is a conversion function from the state enum to character string
*           to be used by logging functions
*
*****************************************************************************/
char* pcStateToString( uint8_t ucState )
{
    char* pResult = NULL;

    switch( ucState )
    {
    case SMBUS_STATE_INITIAL:
        pResult = pcStateSMBusStateInitial;
        break;

    case SMBUS_STATE_AWAITING_COMMAND_BYTE:
        pResult = pcStateSMBusStateAwaitingCommandByte;
        break;

    case SMBUS_STATE_AWAITING_BLOCK_SIZE:
        pResult = pcStateSMBusStateAwaitingBlockSize;
        break;

    case SMBUS_STATE_AWAITING_DATA:
        pResult = pcStateSMBusStateAwaitingData;
        break;

    case SMBUS_STATE_AWAITING_READ:
        pResult = pcStateSMBusStateAwaitingRead;
        break;

    case SMBUS_STATE_READY_TO_SEND_BYTE:
        pResult = pcStateSMBusStateReadyToSendByte;
        break;

    case SMBUS_STATE_CHECK_IF_PEC_REQUIRED:
        pResult = pcStateSMBusStateCheckIfPecRequired;
        break;

    case SMBUS_STATE_AWAITING_DONE:
        pResult = pcStateSMBusStateAwaitingDone;
        break;

    case SMBUS_STATE_CONTROLLER_SEND_COMMAND:
        pResult = pcStateSMBusStateControllerSendCommand;
        break;

    case SMBUS_STATE_CONTROLLER_SEND_READ_START:
        pResult = pcStateSMBusStateControllerSendReadStart;
        break;

    case SMBUS_STATE_CONTROLLER_READ_BLOCK_SIZE:
        pResult = pcStateSMBusStateControllerReadBlockSize;
        break;

    case SMBUS_STATE_CONTROLLER_READ_BYTE:
        pResult = pcStateSMBusStateControllerReadByte;
        break;

    case SMBUS_STATE_CONTROLLER_READ_PEC:
        pResult = pcStateSMBusStateControllerReadPec;
        break;

    case SMBUS_STATE_CONTROLLER_READ_DONE:
        pResult = pcStateSMBusStateControllerReadDone;
        break;

    case SMBUS_STATE_CONTROLLER_WRITE_BYTE:
        pResult = pcStateSMBusStateControllerWriteByte;
        break;

    default:
        pResult = pStateUnknown;
        break;
    }

    return pResult;
}

/******************************************************************************
*
* @brief    Log an error when an event the FSM does not expect arrives
*
*****************************************************************************/
static void vSMBusLogUnexpected( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    if( NULL != pxSMBusInstance )
    {
        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                        pxSMBusInstance->xState, ucAnyEvent );
    }
}

/******************************************************************************
*
* @brief    Move current state to previous state and set the newstate as current
*
*****************************************************************************/
static void vSMBusNextStateDecoder( SMBUS_INSTANCE_TYPE* pxSMBusInstance, SMBus_State_Type xNewState )
{
    if( NULL != pxSMBusInstance )
    {
        pxSMBusInstance->xPreviousState = pxSMBusInstance->xState;
        pxSMBusInstance->xState = xNewState;
    }
}

/******************************************************************************
*
* @brief    Perform actions on receiving an Prepare To ARP command
*           Clear AR flag
*
*****************************************************************************/
static void vSMBusARPPrepareToARP( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        /* Clear the AR Flag */
        pxSMBusInstance->ucARFlag = SMBUS_FALSE;
        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                        pxSMBusInstance->ucSMBusAddress, __LINE__ );
        /* No change to AV Flag */
    }
}

/******************************************************************************
*
* @brief    Perform actions on receiving an ARP reset device command
*           AV and AR flags will be set dependent on the ARP capabilities
*           of the instance
*
*****************************************************************************/
static void vSMBusARPResetDevice( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        /* Clear the AR Flag */
        pxSMBusInstance->ucARFlag = SMBUS_FALSE;
        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                        pxSMBusInstance->ucSMBusAddress, __LINE__ );

        if( ( SMBUS_ARP_NON_ARP_CAPABLE != pxSMBusInstance->xARPCapability ) &&
            ( SMBUS_ARP_CAPABILITY_UNKNOWN != pxSMBusInstance->xARPCapability ) )
        {
            if( SMBUS_UDID_FIXED_ADDRESS == ( pxSMBusInstance->ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE] & SMBUS_UDID_ADDRESS_TYPE_MASK ) )
            {
                /* DTA */
                pxSMBusInstance->ucAVFlag = SMBUS_TRUE;
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucSMBusAddress, __LINE__ );
            }
            else if( SMBUS_UDID_DYNAMIC_AND_PERSISTENT == ( pxSMBusInstance->ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE] & SMBUS_UDID_ADDRESS_TYPE_MASK ) )
            {
                /* non-PTA */
                if( SMBUS_TRUE == pxSMBusInstance->ucAVFlag )
                {
                    /* Disable the HW, to take this address off the bus */
                    vSMBusHWWriteTgtControlEnable( pxSMBusInstance->pxSMBusProfile, pxSMBusInstance->ucUDIDMatchedInstance, 0 );
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->ucSMBusAddress, __LINE__ );
                }

                pxSMBusInstance->ucAVFlag = SMBUS_FALSE;
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucSMBusAddress, __LINE__ );
            }
            /* DYNAMIC_AND_PERSISTENT we don't care */
        }
    }
}

/******************************************************************************
*
* @brief    Write a descriptor to the IP to send a NACK
*
*****************************************************************************/
static void vSMBusSendNACK( SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    if( NULL != pxSMBusInstance )
    {
        ucSMBusTargetWriteDescriptorNACK( pxSMBusInstance->pxSMBusProfile );
        pxSMBusInstance->ucNackSent = SMBUS_TRUE;
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is Initial
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateInitial( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_TARGET_WRITE_IRQ:
            if( SMBUS_FALSE == pxSMBusInstance->ucSimpleDevice )
            {
                if( SMBUS_TRUE == pxSMBusInstance->ulI2CDevice )
                {
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
                }
                else
                {
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_COMMAND_BYTE );
                }
            }
            else
            {
                /* For a simple device a WRITE_IRQ can only be SMBUS_PROTOCOL_SEND_BYTE */
                pxSMBusInstance->xProtocol = SMBUS_PROTOCOL_SEND_BYTE;
                pxSMBusInstance->usExpectedByteCount = 1;
                pxSMBusInstance->ucExpectedByteCountPart = 1;
                vSMBusHWWriteRxFifoFillThresholdFillThresh( pxSMBusInstance->pxSMBusProfile,
                                                                    pxSMBusInstance->usExpectedByteCount );

                if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS == ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                {
                    pxSMBusInstance->usDescriptorsSent++;
                }

                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
            }
            break;

        case E_TARGET_READ_IRQ:

            if( SMBUS_FALSE == pxSMBusInstance->ucSimpleDevice )
            {
                if( SMBUS_TRUE == pxSMBusInstance->ulI2CDevice )
                {
                    vSMBusHandleActionGetI2CDataFromApplication( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_READY_TO_SEND_BYTE );
                }
                else
                {
                    /* although this is unexpected, transmit data byte to avoid hanging the bus */
                    ucSMBusTargetReadDescriptorRead( pxSMBusInstance->pxSMBusProfile, SMBUS_UNEXPECTED_READ_DESCRIPTOR_DATA );
                    vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
                    vSMBusHandleActionResetAllData( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
                }
            }
            else
            {
                /* For a simple device a READ_IRQ can only be SMBUS_PROTOCOL_RECEIVE_BYTE */
                pxSMBusInstance->xProtocol = SMBUS_PROTOCOL_RECEIVE_BYTE;
                vSMBusHandleActionGetDataFromApplication( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_READY_TO_SEND_BYTE );
            }
            break;

        case E_SEND_NEXT_BYTE:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            ulSMBusHWReadPHYCtrlDbgState( pxTheProfile ), __LINE__ );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            ulSMBusHWReadCtrlDbgState( pxTheProfile ), __LINE__ );

            /* Log Controller Message Start */
            if( SMBUS_PROTOCOL_NONE != pxSMBusInstance->xProtocol )
            {
                pxSMBusInstance->ulMessagesInitiated[pxSMBusInstance->xProtocol]++;
            }

            switch( pxSMBusInstance->xProtocol )
            {
                case SMBUS_PROTOCOL_QUICK_COMMAND_LO:
                    ucSMBusControllerWriteDescriptorQuickWrite( pxSMBusInstance->pxSMBusProfile,
                                                                pxSMBusInstance->ucSMBusDestinationAddress );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, 0x1 );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                    break;

                case SMBUS_PROTOCOL_QUICK_COMMAND_HI:
                    ucSMBusControllerReadDescriptorQuickRead( pxSMBusInstance->pxSMBusProfile,
                                                                pxSMBusInstance->ucSMBusDestinationAddress );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, 0x1 );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                    break;

                case SMBUS_PROTOCOL_RECEIVE_BYTE:
                    pxSMBusInstance->usExpectedByteCount = 1;
                    pxSMBusInstance->ucExpectedByteCountPart = 1;
                    vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_SEND_READ_START );
                    break;

                case I2C_PROTOCOL_WRITE:
                case I2C_PROTOCOL_WRITE_READ:
                    ucSMBusControllerWriteDescriptorStartWrite( pxSMBusInstance->pxSMBusProfile,
                                                                    pxSMBusInstance->ucSMBusDestinationAddress );
                    vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                    break;

                case I2C_PROTOCOL_READ:
                    ucSMBusControllerReadDescriptorStart( pxSMBusInstance->pxSMBusProfile,
                                                                    pxSMBusInstance->ucSMBusDestinationAddress );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                    pxSMBusInstance->ucExpectedByteCountPart =
                        ( pxSMBusInstance->usExpectedByteCount < SMBUS_HALF_FIFO_DEPTH ? pxSMBusInstance->usExpectedByteCount : SMBUS_HALF_FIFO_DEPTH );

                    /* Set to no more than 32, half the depth of the FIFO */
                    if( 0 < pxSMBusInstance->ucExpectedByteCountPart ) /* If its zero leave threshold at 1 */
                    {
                        vSMBusHWWriteCtrlRxFifoFillThreshold( pxSMBusInstance->pxSMBusProfile,
                                                                    pxSMBusInstance->ucExpectedByteCountPart );
                    }
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_BYTE );
                    break;

                default:
                    /* Tell the IP to send the WRITE_START */
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->ucSMBusDestinationAddress, __LINE__ );
                    ucSMBusControllerWriteDescriptorStartWrite( pxSMBusInstance->pxSMBusProfile,
                                                                    pxSMBusInstance->ucSMBusDestinationAddress );
                    vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_SEND_COMMAND );
                    break;
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    This function swaps the locations of two SMBus instance numbers
*           within an array of SMBus instances
*
*****************************************************************************/
static void vInstanceSwap( uint8_t* pucInstanceA, uint8_t* pucInstanceB )
{
    if( ( NULL != pucInstanceA ) && ( NULL != pucInstanceB ) )
    {
        uint8_t ucTempInstance = *pucInstanceA;
        *pucInstanceA = *pucInstanceB;
        *pucInstanceB = ucTempInstance;
    }
}

/******************************************************************************
*
* @brief    This function takes an array of SMBus instances
*           It walks through that array and reorders the instances
*           in the order that they should respond to a ARP GetUDID general command
*
*****************************************************************************/
static void vUDIDInstanceSort( SMBUS_PROFILE_TYPE* pxTheProfile, uint8_t pucInstanceIDArray[], uint8_t ucArraySize , uint8_t ucUDIDByteIndex )
{
    uint8_t ucOuterArrayElement = 0;
    uint8_t ucInnerArrayElement = 0;
    uint8_t ucSmallestElement   = 0;

    if( NULL != pxTheProfile )
    {
        for( ucOuterArrayElement = 0; ucOuterArrayElement < ucArraySize - 1; ucOuterArrayElement++ )
        {
            /* Find the minimum element in unsorted array */
            ucSmallestElement = ucOuterArrayElement;
            for( ucInnerArrayElement = ucOuterArrayElement + 1; ucInnerArrayElement < ucArraySize; ucInnerArrayElement++ )
            {
                if( pxTheProfile->xSMBusInstance[pucInstanceIDArray[ucInnerArrayElement]].ucUDID[ucUDIDByteIndex] <
                    pxTheProfile->xSMBusInstance[pucInstanceIDArray[ucSmallestElement]].ucUDID[ucUDIDByteIndex] )
                {
                    ucSmallestElement = ucInnerArrayElement;
                }
            }
            /* Swap the instance of the smallest found element with the instance of the first element */
            vInstanceSwap( &pucInstanceIDArray[ucSmallestElement], &pucInstanceIDArray[ucOuterArrayElement] );
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingCommandByte
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingCommandByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_INSTANCE_TYPE*    pxMatchedInstance                                   = NULL;
    SMBUS_PROFILE_TYPE*     pxTheProfile                                        = NULL;
    uint8_t                 ucOKToACK                                           = SMBUS_FALSE;
    int                     i                                                   = 0;
    int                     j                                                   = 0;
    SMBUS_INSTANCE_TYPE*    pxInstanceJ                                         = NULL;
    SMBUS_INSTANCE_TYPE*    pxInstanceJPlus                                     = NULL;
    uint8_t                 ucUdidListSize                                      = 0;
    uint8_t                 ucUdidList[SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES] = { 0 };

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_DESC_FIFO_ALMOST_EMPTY_IRQ:
            break;

        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_TARGET_DATA_IRQ:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->ucSMBusAddress, __LINE__ );
            if( SMBUS_NOTIFY_ARP_MASTER_ADDRESS == pxSMBusInstance->ucSMBusAddress ) /* Notify ARP Master Instance */
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucSMBusAddress, __LINE__ );
                pxSMBusInstance->xProtocol = SMBUS_PROTOCOL_HOST_NOTIFY;
                pxSMBusInstance->ucCommand = ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                pxSMBusInstance->ucSMBusSenderAddress = pxSMBusInstance->ucCommand;
                pxSMBusInstance->usExpectedByteCount = 2;
                pxSMBusInstance->ucExpectedByteCountPart = 2;
                if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                {
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                    0, __LINE__ );
                }
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
            }
            else if( SMBUS_DEVICE_DEFAULT_ARP_ADDRESS == pxSMBusInstance->ucSMBusAddress ) /* ARP Instance */
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucSMBusAddress, __LINE__ );
                if( SMBUS_ACTION_ARP_PROTOCOL_UNDETERMINED == ucSMBusHandleActionGetARPProtocol( pxSMBusInstance ) )
                {
                    vSMBusSendNACK( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                }
                else
                {
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->xProtocol, __LINE__ );

                    if( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS == pxSMBusInstance->xProtocol )
                    {
                        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                        {
                            pxTheProfile->ucUDIDMatch[i] = SMBUS_TRUE;
                        }

                        if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                        {
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                            0, __LINE__ );
                        }
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_BLOCK_SIZE );
                    }

                    if( ( SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP         == pxSMBusInstance->xProtocol ) ||
                        ( SMBUS_ARP_PROTOCOL_RESET_DEVICE           == pxSMBusInstance->xProtocol ) ||
                        ( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED  == pxSMBusInstance->xProtocol ) )
                    {
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                        pxSMBusInstance->xProtocol, __LINE__ );
                        pxSMBusInstance->usExpectedByteCount = 0;
                        pxSMBusInstance->ucExpectedByteCountPart = 0;

                        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                        {
                            /* Check that at least one target is ARP capable */
                            pxMatchedInstance = &( pxTheProfile->xSMBusInstance[i] );
                            if( ( SMBUS_ARP_CAPABLE                 == pxMatchedInstance->xARPCapability ) ||
                                ( SMBUS_ARP_FIXED_AND_DISCOVERABLE  == pxMatchedInstance->xARPCapability ) )
                            {
                                /* If the ARP controller is on the same IP don't change it's settings */
                                if( pxTheProfile->ucInstanceInPlay != i )
                                {
                                    ucOKToACK = SMBUS_TRUE;
                                    break;
                                }
                            }
                        }

                        /* ACK the command */
                        if( SMBUS_TRUE == ucOKToACK )
                        {
                            if( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED  == pxSMBusInstance->xProtocol )
                            {
                                ucOKToACK = SMBUS_FALSE;
                                for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                                {
                                    /* Find the instance corresponding to the directed address */
                                    pxMatchedInstance = &( pxTheProfile->xSMBusInstance[i] );
                                    if( pxMatchedInstance->ucSMBusAddress == pxSMBusInstance->ucMatchedSMBusAddress )
                                    {
                                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                        pxMatchedInstance->ucAVFlag, __LINE__ );
                                        /* Now for that instance, check if its AV flag is set */
                                        if( SMBUS_TRUE == pxMatchedInstance->ucAVFlag )
                                        {
                                            ucOKToACK = SMBUS_TRUE;
                                        }
                                        break;
                                    }
                                }

                                if( SMBUS_TRUE == ucOKToACK )
                                {
                                    /* ACK the command */
                                    if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                                    {
                                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                                        0, __LINE__ );
                                    }
                                    ucSMBusTargetWriteDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                                    pxSMBusInstance->ucPECSent = SMBUS_TRUE;
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                                }
                                else
                                {
                                    /* NACK the command */
                                    vSMBusSendNACK( pxSMBusInstance );
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                                }
                            }
                            else
                            {
                                /* ACK the command */
                                if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                                {
                                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                                    0, __LINE__ );
                                }
                                ucSMBusTargetWriteDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                                pxSMBusInstance->ucPECSent = SMBUS_TRUE;
                                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                            }
                        }
                        else
                        {
                            /* NACK the command */
                            vSMBusSendNACK( pxSMBusInstance );
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                        }
                    } /* SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP, SMBUS_ARP_PROTOCOL_RESET_DEVICE and SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED */

                    if( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED == pxSMBusInstance->xProtocol )
                    {
                        pxSMBusInstance->usSendDataSize = SMBUS_GET_UDID_MSG_LENGTH;
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->ucMatchedSMBusAddress, __LINE__ );

                        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                        {
                            pxMatchedInstance = &( pxTheProfile->xSMBusInstance[i] );

                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxMatchedInstance->ucSMBusAddress, __LINE__ );

                            /* If we have a matched address and it is ARP capable */
                            if( pxMatchedInstance->ucSMBusAddress == pxSMBusInstance->ucMatchedSMBusAddress )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxMatchedInstance->xARPCapability, __LINE__ );
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxMatchedInstance->ucAVFlag, __LINE__ );

                                if( SMBUS_ARP_NON_ARP_CAPABLE != pxMatchedInstance->xARPCapability )
                                {
                                    if( SMBUS_TRUE == pxMatchedInstance->ucAVFlag )
                                    {
                                        ucOKToACK = SMBUS_TRUE;

                                        /* Byte Count */
                                        pxSMBusInstance->ucSendData[0] = SMBUS_GET_UDID_DATA_LENGTH;

                                        /* Copy the UDID into send buffer */
                                        for( j = 0; j < SMBUS_UDID_LENGTH; j++ )
                                        {
                                            pxSMBusInstance->ucSendData[j + 1] = pxMatchedInstance->ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE - j];
                                        }

                                        /* Device Slave Address - Bit 0 in Slave Address Field should be a 1 */
                                        pxSMBusInstance->ucSendData[SMBUS_UDID_ASSIGNED_ADDRESS_BYTE] =
                                            ( ( pxMatchedInstance->ucSMBusAddress << 1 ) | SMBUS_UDID_ASSIGNED_ADDRESS_BIT0 );
                                        break;
                                    }
                                }
                            }
                        }
                        if( SMBUS_TRUE == ucOKToACK )
                        {
                            /* ACK the command */
                            if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR, 0, __LINE__ );
                            }
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_READ );
                        }
                        else
                        {
                            /* NACK the command */
                            vSMBusSendNACK( pxSMBusInstance );
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                        }
                    } /* SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED */

                    if( SMBUS_ARP_PROTOCOL_GET_UDID == pxSMBusInstance->xProtocol )
                    {
                        pxSMBusInstance->usSendDataSize = SMBUS_GET_UDID_MSG_LENGTH;

                        /* Look up UUID to send */
                        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                        {
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG, i, __LINE__ );
                                            pxMatchedInstance = &( pxTheProfile->xSMBusInstance[i] );
                            if( SMBUS_TRUE == pxMatchedInstance->ucInstanceInUse )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxMatchedInstance->xARPCapability, __LINE__ );
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxMatchedInstance->ucARFlag, __LINE__ );
                                                /* If we have a matched address and it is ARP capable */
                                if( ( SMBUS_ARP_CAPABLE                 == pxMatchedInstance->xARPCapability ) ||
                                    ( SMBUS_ARP_FIXED_AND_DISCOVERABLE  == pxMatchedInstance->xARPCapability ) )
                                {
                                    /* If the ARP controller is on the same IP don't change it's settings */
                                    if( pxTheProfile->ucInstanceInPlay != i )
                                    {
                                        if( SMBUS_FALSE == pxMatchedInstance->ucARFlag )
                                        {
                                            /* Add this instance to a list */
                                            ucUdidList[ucUdidListSize++] = i;

                                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                    ucUdidListSize, __LINE__ );
                                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                    i, __LINE__ );
                                        }
                                    }
                                }
                            }
                        }

                        /* Now go through the list and eliminate the keep instances with the smallest UDID */
                        /* Start with UDID byte 15 and keep iterating til only 1 instance remains */
                        if( SMBUS_ZERO_ELEMENTS < ucUdidListSize )
                        {
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                ucUdidListSize, __LINE__ );
                            if( SMBUS_SINGLE_ELEMENT != ucUdidListSize )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                ucUdidListSize, __LINE__ );
                                for( i = SMBUS_UDID_DEVICE_CAPABILITIES_BYTE; i >= 0; i-- )
                                {
                                    vUDIDInstanceSort( pxTheProfile, ucUdidList, ucUdidListSize, i );

                                    /* Check if 2 or more elements are equal and remove array instances that are greater than the minimum */
                                    for( j = 0; j < ucUdidListSize; j++ )
                                    {
                                        pxInstanceJ     = &( pxTheProfile->xSMBusInstance[ucUdidList[j]] );
                                        pxInstanceJPlus = &( pxTheProfile->xSMBusInstance[ucUdidList[j+1]] );

                                        if ( pxInstanceJ->ucUDID[i] != pxInstanceJPlus->ucUDID[i] )
                                        {
                                            /* The two bytes don't match so we can break here */
                                            ucUdidListSize = j + 1;
                                            break;
                                        }
                                    }

                                    /* Break out of the loop as soon as we have determined the smallest UDID ie. array size is 1 */
                                    if ( SMBUS_SINGLE_ELEMENT == ucUdidListSize )
                                    {
                                        break;
                                    }
                                }
                            }

                            /* Now respond with the UDID of the instance identified */
                            pxMatchedInstance = &( pxTheProfile->xSMBusInstance[ucUdidList[0]] );

                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            ucUdidList[0], SMBUS_LOG_EVENT_DEBUG,
                                            pxMatchedInstance->ucSMBusAddress, __LINE__ );

                            ucOKToACK = SMBUS_TRUE;

                            /* Byte Count */
                            pxSMBusInstance->ucSendData[0] = SMBUS_GET_UDID_DATA_LENGTH;

                            /* Copy the UDID into send buffer */
                            for( j = 0; j < SMBUS_UDID_LENGTH; j++ )
                            {
                                pxSMBusInstance->ucSendData[j + 1] = pxMatchedInstance->ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE - j];
                            }

                            /* Device Slave Address */
                            /* If AV flag is clear then Data 17 field should be all 1s */
                            if( SMBUS_FALSE == pxMatchedInstance->ucAVFlag )
                            {
                                pxSMBusInstance->ucSendData[SMBUS_UDID_ASSIGNED_ADDRESS_BYTE] = SMBUS_UDID_DTA_AV_FLAG_CLEAR;
                            }
                            else
                            {
                                /* Bit 0 in Slave Address Field should be a 1 */
                                pxSMBusInstance->ucSendData[SMBUS_UDID_ASSIGNED_ADDRESS_BYTE] =
                                    ( ( pxMatchedInstance->ucSMBusAddress << 1 ) | SMBUS_UDID_DTA_BIT_0_SET );
                            }
                        }

                        if( SMBUS_TRUE == ucOKToACK )
                        {
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                            pxMatchedInstance->ucSMBusAddress, __LINE__ );
                            /* ACK the command */
                            if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                                0, __LINE__ );
                            }
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_READ );
                        }
                        else
                        {
                            vSMBusSendNACK( pxSMBusInstance );
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                        }
                    } /* SMBUS_ARP_PROTOCOL_GET_UDID */
                }
            }
            else /* SMBus Devices */
            {
                /* Need protocol now. */
                vSMBusHandleActionGetProtocol( pxSMBusInstance );

                switch( pxSMBusInstance->xProtocol )
                {
                case SMBUS_PROTOCOL_BLOCK_READ:
                case SMBUS_PROTOCOL_READ_64:
                case SMBUS_PROTOCOL_READ_32:
                case SMBUS_PROTOCOL_READ_WORD:
                case SMBUS_PROTOCOL_READ_BYTE:
                    vSMBusHandleActionGetDataFromApplication( pxSMBusInstance );
                    if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                    {
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                        0, __LINE__ );
                    }
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_READ );
                    break;

                case SMBUS_PROTOCOL_WRITE_64:
                    pxSMBusInstance->usExpectedByteCount = 8;
                    pxSMBusInstance->ucExpectedByteCountPart = 8;
                    break;

                case SMBUS_PROTOCOL_WRITE_32:
                    pxSMBusInstance->usExpectedByteCount = 4;
                    pxSMBusInstance->ucExpectedByteCountPart = 4;
                    break;

                case SMBUS_PROTOCOL_WRITE_WORD:

                case SMBUS_PROTOCOL_PROCESS_CALL:
                    pxSMBusInstance->usExpectedByteCount = 2;
                    pxSMBusInstance->ucExpectedByteCountPart = 2;
                    break;

                case SMBUS_PROTOCOL_WRITE_BYTE:
                    pxSMBusInstance->usExpectedByteCount = 1;
                    pxSMBusInstance->ucExpectedByteCountPart = 1;
                    break;

                case SMBUS_PROTOCOL_BLOCK_WRITE:
                case SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL:
                    if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                    {
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_ERROR,
                                        0, __LINE__ );
                    }
                    /* Don't increment Descriptors_Sent. We'll do that once we know the block size. */
                    /* pxSMBusInstance->Descriptors_Sent++; */
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_BLOCK_SIZE );
                    break;

                default:
                    /* If Protocol is not recognized then NACK */
                    vSMBusSendNACK( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                    break;
                }

                switch( pxSMBusInstance->xProtocol )
                {
                case SMBUS_PROTOCOL_WRITE_64:
                case SMBUS_PROTOCOL_WRITE_32:
                case SMBUS_PROTOCOL_WRITE_WORD:
                case SMBUS_PROTOCOL_WRITE_BYTE:
                case SMBUS_PROTOCOL_PROCESS_CALL:
                    vSMBusHWWriteRxFifoFillThresholdFillThresh( pxSMBusInstance->pxSMBusProfile,
                                                                        pxSMBusInstance->usExpectedByteCount );
                    for( i = 0; i < pxSMBusInstance->usExpectedByteCount + 1; i++ )
                    {
                        if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS == ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                        {
                            pxSMBusInstance->usDescriptorsSent++;
                        }
                    }

                    /* Check if more data is already waiting */
                    if( SMBUS_RX_FIFO_IS_EMPTY != ulSMBusHWReadTgtRxFifoStatusEmpty( pxSMBusInstance->pxSMBusProfile ) )
                    {
                        /* If FIFO isn't empty generate an E_TARGET_DATA_IRQ event
                           and handle in SMBUS_STATE_AWAITING_DATA */
                        vSMBusGenerateEvent_E_TARGET_DATA_IRQ( pxSMBusInstance );
                    }

                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
                    break;

                default:
                    break;
                }
            }

            if( SMBUS_PROTOCOL_NONE != pxSMBusInstance->xProtocol )
            {
                pxSMBusInstance->ulMessagesInitiated[pxSMBusInstance->xProtocol]++;
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingRead
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingRead( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;
    uint8_t             ucCurrentFill = 0;
    int                 i             = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_DESC_FIFO_ALMOST_EMPTY_IRQ:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_INFO,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usDescriptorsSent, __LINE__ );

            ucCurrentFill = ulSMBusHWReadTgtDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
            for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
            {
                if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
                {
                    uint8_t ucNoStatusCheck = SMBUS_TRUE;
                    if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                    {
                        break;
                    }
                    else
                    {
                        pxSMBusInstance->usDescriptorsSent++;
                    }
                }
            }
            break;

        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case    E_TARGET_READ_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_64                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) )
            {
                /* vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance ); */
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_READY_TO_SEND_BYTE );
            }
            break;

        case E_TARGET_DATA_IRQ:
             /* DATA_IRQ may have been added to the queue at the same time as READ_IRQ */
             /* Ignore for now DATA_IRQ will assert again */
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ReadyToSendByte
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateReadyToSendByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile  = NULL;
    uint8_t             ucCurrentFill = 0;
    int                 i             = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case    E_DESC_FIFO_ALMOST_EMPTY_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_64                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_RECEIVE_BYTE                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_TRUE                                            == pxSMBusInstance->ulI2CDevice ) )
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usSendIndex, __LINE__ );
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usSendDataSize, __LINE__ );

                /* Read how much space is available in the descriptor FIFO and write that number of ACKs */
                ucCurrentFill = ulSMBusHWReadTgtDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                {
                    if( pxSMBusInstance->usSendIndex < pxSMBusInstance->usSendDataSize )
                    {
                        if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusTargetReadDescriptorRead( pxSMBusInstance->pxSMBusProfile,
                                                                pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex] ) )
                        {
                            break;
                        }
                        else
                        {
                            pxSMBusInstance->usSendIndex++;
                        }
                    }
                }

                if( pxSMBusInstance->usSendIndex == pxSMBusInstance->usSendDataSize )
                {
                    vSMBusHandleActionCreateEventIsPECRequired( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CHECK_IF_PEC_REQUIRED );
                }
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is CheckIfPECRequired
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateCheckIfPECRequired( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_IS_PEC_REQUIRED:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber,
                            SMBUS_LOG_EVENT_DEBUG, pxSMBusInstance->xProtocol, __LINE__ );
            if( ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_64                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_RECEIVE_BYTE                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_TRUE                                            == pxSMBusInstance->ulI2CDevice ) )
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucPECRequired, __LINE__ );
                if( SMBUS_TRUE == pxSMBusInstance->ucPECRequired )
                {
                    if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusTargetReadDescriptorPECRead( pxSMBusInstance->pxSMBusProfile ) )
                    {
                        vSMBusHandleActionCreateEventIsPECRequired( pxSMBusInstance );
                    }
                    else
                    {
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                    }
                }
                else
                {
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                }
            }

            if( ( SMBUS_ARP_PROTOCOL_GET_UDID           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED  == pxSMBusInstance->xProtocol ) )
            {
                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusTargetReadDescriptorPECRead( pxSMBusInstance->pxSMBusProfile ) )
                {
                    vSMBusHandleActionCreateEventIsPECRequired( pxSMBusInstance );
                }
                else
                {
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                }
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingDone
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingDone( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_INSTANCE_TYPE* pMatched_Instance = NULL;
    SMBUS_PROFILE_TYPE*  pxTheProfile      = NULL;
    uint8_t              ucCurrentFill     = 0;
    int                  i                 = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usDescriptorsSent, __LINE__ );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usExpectedByteCount, __LINE__ );
            break;

        case E_DESC_FIFO_ALMOST_EMPTY_IRQ:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usDescriptorsSent, __LINE__ );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usExpectedByteCount, __LINE__ );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            pxSMBusInstance->usReceiveIndex, __LINE__ );

            pxSMBusInstance->ucFifoEmptyWhileInDoneCount++;

            if( SMBUS_MAX_FIFO_EMPTY_WHILE_IN_DONE < pxSMBusInstance->ucFifoEmptyWhileInDoneCount )
            {
                /* We have an issue */
                vSMBusSendNACK( pxSMBusInstance );
            }
            else
            {
                /* We may have got here as all data has arrived but we may not have sent all ACKs necessary */
                /* However if we NACK in the middle of ASSIGN_ARP don't send more descriptors */
                if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                {
                    ucCurrentFill = ulSMBusHWReadTgtDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                    {
                        if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
                        {
                            uint8_t ucNoStatusCheck = SMBUS_TRUE;
                            if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                        }
                    }
                }

                /* Maybe a PEC arrived */
                if( pxSMBusInstance->usReceiveIndex >= pxSMBusInstance->usExpectedByteCount + 1 )
                {
                    /* Should I have received a PEC */
                    if( ( SMBUS_TRUE == pxSMBusInstance->ucPECRequired ) &&
                        ( SMBUS_FALSE == pxSMBusInstance->ucPECSent ) )
                    {
                        ucSMBusTargetWriteDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                        pxSMBusInstance->ucPECSent = SMBUS_TRUE;
                    }
                    else
                    {
                        /* Send a NACK and wait on DONE */
                        if( SMBUS_TRUE == pxSMBusInstance->ucSimpleDevice )
                        {
                            vSMBusSendNACK( pxSMBusInstance );
                        }
                    }
                }
            }
            break;

        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_TARGET_READ_IRQ:
            /* Should not get this here. Send a NACK and wait on DONE */
            vSMBusSendNACK( pxSMBusInstance );
            break;

        case E_TARGET_DONE_IRQ:
            if( SMBUS_PROTOCOL_NONE == pxSMBusInstance->xProtocol )
            {
                /* VMC_ERR( "ERROR: Protocol = SMBUS_PROTOCOL_NONE\n" ); */
            }
            else
            {
                pxSMBusInstance->ulMessagesComplete[pxSMBusInstance->xProtocol]++;
            }

            /* I2C Target has no protocol set so use ulI2CDevice */
            if( SMBUS_TRUE == pxSMBusInstance->ulI2CDevice )
            {
                /* Report Data to Application */
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucCommand, __LINE__ );
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usExpectedByteCount, __LINE__ );

                if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                {
                    vSMBusHandleActionAnnounceI2CResultToApplication( pxSMBusInstance, SMBUS_SUCCESS );
                }
                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            else
            {
                if( ( SMBUS_PROTOCOL_BLOCK_WRITE                            == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_64                               == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_32                               == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_WORD                             == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_BYTE                             == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_SEND_BYTE                              == pxSMBusInstance->xProtocol ) )
                {
                    /* Report Data to Application */
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->ucCommand, __LINE__ );
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->usExpectedByteCount, __LINE__ );

                    if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                    {
                        vSMBusHandleActionWriteDataToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID );
                    }
                }

                if( SMBUS_PROTOCOL_HOST_NOTIFY == pxSMBusInstance->xProtocol )
                {
                    /* Report Data to Application */
                    vSMBusHandleActionWriteDataToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID );
                }

                if( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS == pxSMBusInstance->xProtocol )
                {
                    if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                    {
                        /* Change the address for the required slave
                           Determine which slave instance the UDID matched for */
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                            pxSMBusInstance->ucUDIDMatchedInstance, __LINE__ );
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                            pxSMBusInstance->ucNewDeviceSlaveAddress, __LINE__ );

                        pMatched_Instance = &( pxTheProfile->xSMBusInstance[pxSMBusInstance->ucUDIDMatchedInstance] );

                        if( ( SMBUS_ARP_CAPABLE                 == pMatched_Instance->xARPCapability ) ||
                            ( SMBUS_ARP_FIXED_AND_DISCOVERABLE  == pMatched_Instance->xARPCapability ) )
                        {
                            pMatched_Instance->ucARFlag = SMBUS_TRUE;
                            pMatched_Instance->ucAVFlag = SMBUS_TRUE;
                        }

                        if( SMBUS_ARP_CAPABLE == pMatched_Instance->xARPCapability )
                        {
                            pMatched_Instance->ucSMBusAddress = pxSMBusInstance->ucNewDeviceSlaveAddress;

                            /* Disable the HW, Change the address and re-enable the HW */
                            vSMBusHWWriteTgtControlEnable( pxTheProfile, pxSMBusInstance->ucUDIDMatchedInstance, 0 );
                            vSMBusHWWriteTgtControlAddress( pxTheProfile, pxSMBusInstance->ucUDIDMatchedInstance,
                                                                pMatched_Instance->ucSMBusAddress );
                            vSMBusHWWriteTgtControlEnable( pxTheProfile, pxSMBusInstance->ucUDIDMatchedInstance, 1 );

                            /* Report Data to Application */
                            vSMBusHandleActionNotifyAddressChangeToApplication( pMatched_Instance,
                                                                                        pxTheProfile->ulTransactionID );
                        }
                    }
                }

                if( SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP == pxSMBusInstance->xProtocol )
                {
                    for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                    {
                        pMatched_Instance = &( pxTheProfile->xSMBusInstance[i] );
                        if( SMBUS_TRUE == pMatched_Instance->ucInstanceInUse )
                        {
                            if( ( SMBUS_ARP_CAPABLE                 == pMatched_Instance->xARPCapability ) ||
                                ( SMBUS_ARP_FIXED_AND_DISCOVERABLE  == pMatched_Instance->xARPCapability ) ||
                                ( SMBUS_ARP_FIXED_NOT_DISCOVERABLE  == pMatched_Instance->xARPCapability ) )
                            {
                                /* If the ARP controller is on the same IP don't change it's settings */
                                if( pxTheProfile->ucInstanceInPlay != i )
                                {
                                    vSMBusARPPrepareToARP( pMatched_Instance );
                                }
                            }
                        }
                    }
                }

                if( SMBUS_ARP_PROTOCOL_RESET_DEVICE == pxSMBusInstance->xProtocol )
                {
                    for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                    {
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG, i, __LINE__ );
                        pMatched_Instance = &( pxTheProfile->xSMBusInstance[i] );
                        if( SMBUS_TRUE == pMatched_Instance->ucInstanceInUse )
                        {
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                            pMatched_Instance->ucSMBusAddress, __LINE__ );
                            if( ( SMBUS_ARP_CAPABLE                == pMatched_Instance->xARPCapability ) ||
                                ( SMBUS_ARP_FIXED_AND_DISCOVERABLE == pMatched_Instance->xARPCapability ) ||
                                ( SMBUS_ARP_FIXED_NOT_DISCOVERABLE == pMatched_Instance->xARPCapability ) )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pMatched_Instance->ucSMBusAddress, __LINE__ );

                                /* If the ARP controller is on the same IP don't change it's settings */
                                if( pxTheProfile->ucInstanceInPlay != i )
                                {
                                    vSMBusARPResetDevice( pMatched_Instance );
                                }
                            }
                        }
                    }
                }

                if( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED == pxSMBusInstance->xProtocol )
                {
                    for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                    {
                        pMatched_Instance = &( pxTheProfile->xSMBusInstance[i] );
                        if( pMatched_Instance->ucSMBusAddress == pxSMBusInstance->ucMatchedSMBusAddress )
                        {
                            if( SMBUS_TRUE == pMatched_Instance->ucInstanceInUse )
                            {
                                if( ( SMBUS_ARP_CAPABLE                == pMatched_Instance->xARPCapability ) ||
                                    ( SMBUS_ARP_FIXED_AND_DISCOVERABLE == pMatched_Instance->xARPCapability ) ||
                                    ( SMBUS_ARP_FIXED_NOT_DISCOVERABLE == pMatched_Instance->xARPCapability ) )
                                {
                                    vSMBusARPResetDevice( pMatched_Instance );
                                }
                            }
                        }
                    }
                }

                if( ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_BLOCK_WRITE                            == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_READ_64                                == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_READ_32                                == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_READ_WORD                              == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_READ_BYTE                              == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_RECEIVE_BYTE                           == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_64                               == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_32                               == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_WORD                             == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_WRITE_BYTE                             == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_SEND_BYTE                              == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_HOST_NOTIFY                            == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_PROTOCOL_NONE                                   == pxSMBusInstance->xProtocol ) )
                {
                    if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                    {
                        vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_SUCCESS );
                    }
                    else
                    {
                        vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
                    }
                    vSMBusHandleActionResetAllData( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
                }

                if( ( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS                     == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP                     == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_ARP_PROTOCOL_RESET_DEVICE                       == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED              == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                    ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) )
                {
                    vSMBusHandleActionResetAllData( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
                }
            }
            break;

        /* Awaiting DONE but a DATA_IRQ arrives means a PEC has arrived */
        case E_TARGET_DATA_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_WRITE                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_64                   == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_32                   == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_WORD                 == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_BYTE                 == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_SEND_BYTE                  == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS         == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP         == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_RESET_DEVICE           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED  == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID               == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED      == pxSMBusInstance->xProtocol ) )
            {
                /* Read as much data as is available */
                ucCurrentFill = ulSMBusHWReadTgtRxFifoStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                for( i = 0; i < ucCurrentFill; i++ )
                {
                    pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex++] =
                        ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                }
            }
            break;

        case E_CONTROLLER_DONE_IRQ:
            if( SMBUS_PROTOCOL_BLOCK_READ == pxSMBusInstance->xProtocol )
            {
                /* Report Data to Application */
                vSMBusHandleActionWriteDataToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID );
            }

            if( SMBUS_PROTOCOL_NONE == pxSMBusInstance->xProtocol )
            {
                /* VMC_ERR( "ERROR: Protocol = SMBUS_PROTOCOL_NONE\n" ); */
            }
            else
            {
                pxSMBusInstance->ulMessagesComplete[pxSMBusInstance->xProtocol]++;
            }

            if( I2C_PROTOCOL_WRITE == pxSMBusInstance->xProtocol )
            {
                vSMBusHandleActionAnnounceI2CResultToApplication( pxSMBusInstance, SMBUS_SUCCESS );
            }
            else
            {
                vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance,
                                                            pxTheProfile->ulTransactionID, SMBUS_SUCCESS );
            }

            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    This function walks through each instance to determine if
*           the UDID transmitted matches any one of the active SMBus instances
*
*****************************************************************************/
static uint8_t ucCheckAtLeastOneMatchFound( SMBUS_PROFILE_TYPE* pxSMBusProfile, SMBUS_INSTANCE_TYPE* pxSMBusInstance )
{
    uint8_t ucAtLeastOneMatchFound = SMBUS_FALSE;
    int     i                      = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( NULL != pxSMBusInstance ) )
    {
        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
        {
            if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
            {
                if( SMBUS_TRUE == pxSMBusProfile->ucUDIDMatch[i] )
                {
                    int Index = 15 - pxSMBusInstance->usReceiveIndex;

                    if( pxSMBusProfile->xSMBusInstance[i].ucUDID[Index]
                            == pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex] )
                    {
                        ucAtLeastOneMatchFound = SMBUS_TRUE;
                        pxSMBusInstance->ucUDIDMatchedInstance = i;
                    }
                    else
                    {
                        pxSMBusProfile->ucUDIDMatch[i] = SMBUS_FALSE;
                    }
                }
            }
        }

        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                        ucAtLeastOneMatchFound, __LINE__ );
    }
    return ( ucAtLeastOneMatchFound );
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingData
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingData( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE*     pxTheProfile        = NULL;
    uint8_t                 ucNoStatusCheck     = SMBUS_TRUE;
    uint8_t                 ucCurrentFill       = 0;
    int                     i                   = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_DESC_FIFO_ALMOST_EMPTY_IRQ:
            /* ACKs need to be done after the received byte is checked for SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS */
            if( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS != pxSMBusInstance->xProtocol )
            {
                if( pxSMBusInstance->ulI2CDevice )
                {
                    /* Read how much space is available in the descriptor FIFO and write that number of ACKs */
                    ucCurrentFill = ulSMBusHWReadTgtDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                    {
                        if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usReceiveIndex )
                        {
                            ucNoStatusCheck = SMBUS_TRUE;
                            if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                        }
                    }
                }
                else
                {
                    /* Read how much space is available in the descriptor FIFO and write that number of ACKs */
                    ucCurrentFill = ulSMBusHWReadTgtDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                    {
                        if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
                        {
                            ucNoStatusCheck = SMBUS_TRUE;
                            if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                        }
                    }
                }
            }
            break;

        case E_TARGET_READ_IRQ:
            if( SMBUS_PROTOCOL_HOST_NOTIFY != pxSMBusInstance->xProtocol )
            {
                if( SMBUS_TRUE == pxSMBusInstance->ulI2CDevice )
                {
                    if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                    {
                        if( 0 < pxSMBusInstance->usReceiveIndex )
                        {
                            vSMBusHandleActionWriteI2CDataToApplication( pxSMBusInstance);
                        }
                    }
                    vSMBusHandleActionGetI2CDataFromApplication( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_READY_TO_SEND_BYTE );
                }
                else
                {
                    /* READ_IRQ  may have been added to the queue along with DATA_IRQ */
                    /* Ignore for now and add back onto the queue */
                    vSMBusGenerateEvent_E_TARGET_READ_IRQ( pxSMBusInstance );
                }
            }
            else
            {
                vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
                vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            break;

        case E_TARGET_DONE_IRQ:
            if( SMBUS_TRUE == pxSMBusInstance->ulI2CDevice )
            {
                /* Received a STOP */
                if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                {
                    if( 0 < pxSMBusInstance->usReceiveIndex )
                    {
                        vSMBusHandleActionWriteI2CDataToApplication( pxSMBusInstance);
                    }
                }
                 /* Not ready to handle this add it back on the queue and handle it in SMBUS_STATE_AWAITING_DONE */
                vSMBusGenerateEvent_E_TARGET_DONE_IRQ( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
            }
            else
            {
                /* Shouldn't get here unless a block write size is incorrect */
                vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
                /* TODO: Need a new error defined since we are calling BusError with DONE_IRQ */
                vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            break;

        case E_TARGET_DATA_IRQ:
            if( SMBUS_TRUE == pxSMBusInstance->ulI2CDevice )
            {
                /* Read as much data as is available */
                ucCurrentFill = ulSMBusHWReadTgtRxFifoStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                for( i = 0; i < ucCurrentFill; i++ )
                {
                    pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex++] =
                        ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                }
            }
            else
            {
                switch( pxSMBusInstance->xProtocol )
                {
                case SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS:
                    {
                        ucCurrentFill = ulSMBusHWReadTgtRxFifoStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                        for( i = 0; i < ucCurrentFill; i++ )
                        {
                            pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex] =
                                ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );

                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->usReceiveIndex, __LINE__ );
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex], __LINE__ );
                            if( SMBUS_UDID_LENGTH > pxSMBusInstance->usReceiveIndex )
                            {
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex], __LINE__ );
                                /* Check if this byte matches one of our instances */
                                if( SMBUS_TRUE == ucCheckAtLeastOneMatchFound( pxSMBusInstance->pxSMBusProfile, pxSMBusInstance ) )
                                {
                                    if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS == ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                                    {
                                        pxSMBusInstance->usDescriptorsSent++;
                                    }
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
                                }
                                else
                                {
                                    vSMBusSendNACK( pxSMBusInstance );
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                                }
                                pxSMBusInstance->usReceiveIndex++;
                            }
                            else if( SMBUS_UDID_LENGTH == pxSMBusInstance->usReceiveIndex )
                            {
                                /* The UDID matched so assign new slave address to that */
                                pxSMBusInstance->ucNewDeviceSlaveAddress =
                                    ( pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex] >> SMBUS_TGT_CONTROL_0_ADDRESS_FIELD_POSITION );

                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex], __LINE__ );
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->ucUDIDMatchedInstance, __LINE__ );
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->ucNewDeviceSlaveAddress, __LINE__ );

                                if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS == ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                                {
                                    pxSMBusInstance->usDescriptorsSent++;
                                }
                                pxSMBusInstance->usReceiveIndex++;
                            }
                            else
                            {
                                /* PEC */
                                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                                pxSMBusInstance->usReceiveIndex, __LINE__ );
                                ucSMBusTargetWriteDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                                pxSMBusInstance->ucPECSent = SMBUS_TRUE;
                                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                            }
                        }
                    }
                    break;

                case SMBUS_PROTOCOL_HOST_NOTIFY:
                case SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL:
                case SMBUS_PROTOCOL_PROCESS_CALL:
                case SMBUS_PROTOCOL_BLOCK_WRITE:
                case SMBUS_PROTOCOL_WRITE_64:
                case SMBUS_PROTOCOL_WRITE_32:
                case SMBUS_PROTOCOL_WRITE_WORD:
                case SMBUS_PROTOCOL_WRITE_BYTE:
                case SMBUS_PROTOCOL_SEND_BYTE:
                    /* Read as much data as is available */
                    ucCurrentFill = ulSMBusHWReadTgtRxFifoStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ucCurrentFill; i++ )
                    {
                        pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex++] =
                            ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                    }

                    /* Check if we need to change threshold level */
                    if( pxSMBusInstance->usReceiveIndex < pxSMBusInstance->usExpectedByteCount )
                    {
                        if( SMBUS_FIFO_FILL_TRIGGER > ( pxSMBusInstance->usExpectedByteCount - pxSMBusInstance->usReceiveIndex ) )
                        {
                            pxSMBusInstance->ucExpectedByteCountPart = 1;
                        }
                        else
                        {
                            pxSMBusInstance->ucExpectedByteCountPart = SMBUS_FIFO_FILL_TRIGGER;
                        }

                        /* Set up the threshold again */
                        vSMBusHWWriteRxFifoFillThresholdFillThresh( pxSMBusInstance->pxSMBusProfile,
                                                                            pxSMBusInstance->ucExpectedByteCountPart );

                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
                    }
                    else /* We've got all the bytes */
                    {
                        /* Set threshold back to 1 */
                        vSMBusHWWriteRxFifoFillThresholdFillThresh( pxSMBusInstance->pxSMBusProfile, 1 );

                        if( ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                            ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) )
                        {
                            /* Write the received data */
                            if( SMBUS_FALSE == pxSMBusInstance->ucNackSent )
                            {
                                vSMBusHandleActionWriteDataToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID );
                            }
                            vSMBusHandleActionGetDataFromApplication( pxSMBusInstance );
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                            pxSMBusInstance->usSendDataSize, __LINE__ );

                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_READ );
                        }
                        else
                        {
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                        }
                    }
                    break;

                default:
                    break;
                }
            break; /* case E_TARGET_DATA_IRQ: */

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
            }
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is AwaitingBlockSize
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateAwaitingBlockSize( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_DESC_FIFO_ALMOST_EMPTY_IRQ:
            break;

        case E_TARGET_DATA_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE                            == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS                     == pxSMBusInstance->xProtocol ) )
            {
                /* Get the block size and set Expected_Byte_Count */
                if( SMBUS_RX_FIFO_IS_EMPTY != ulSMBusHWReadTgtRxFifoStatusEmpty( pxSMBusInstance->pxSMBusProfile ) )
                {
                    pxSMBusInstance->usExpectedByteCount =
                        ulSMBusHWReadTgtRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->usDescriptorsSent, __LINE__ );
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->usExpectedByteCount, __LINE__ );

                    /* Now send ACK for the block size */
                    if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS != ucSMBusTargetWriteDescriptorACK( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE ) )
                    {
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR,
                                        pxSMBusInstance->ucThisInstanceNumber,
                                        SMBUS_LOG_EVENT_ERROR, 0, __LINE__ );
                    }
                }
                else
                {
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    0, __LINE__ );
                    break;
                }

                pxSMBusInstance->usReceiveIndex = 0;

                /* Change threshold to match block size - or full fifo */
                pxSMBusInstance->ucExpectedByteCountPart =
                    ( pxSMBusInstance->usExpectedByteCount < SMBUS_FIFO_FILL_TRIGGER ? 1 : SMBUS_FIFO_FILL_TRIGGER );

                vSMBusHWWriteRxFifoFillThresholdFillThresh( pxSMBusInstance->pxSMBusProfile,
                                                                    pxSMBusInstance->ucExpectedByteCountPart );

                /* Check for block size of 0 */
                if( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL == pxSMBusInstance->xProtocol )
                {
                    if( 0 == pxSMBusInstance->usExpectedByteCount )
                    {
                        /* Now get data to send back */
                        vSMBusHandleActionGetDataFromApplication( pxSMBusInstance );
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                        pxSMBusInstance->usSendDataSize, __LINE__ );
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_READ );
                    }
                    else
                    {
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
                    }
                }
                else
                {
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DATA );
                }
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerSendCommand
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerSendCommand( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_SEND_NEXT_BYTE:
            if( ( SMBUS_PROTOCOL_BLOCK_READ             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_64                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED  == pxSMBusInstance->xProtocol ) )
            {
                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorByte( pxSMBusInstance->pxSMBusProfile,
                                                            pxSMBusInstance->ucCommand, SMBUS_FALSE ) )
                {
                    /* If SMBus_Controller_Write_Descriptor_Byte(  ) is true pop out and wait on descriptor fifo emptying */
                }
                else
                {
                    vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_SEND_READ_START );
                }
            }

            if( ( SMBUS_PROTOCOL_BLOCK_WRITE                            == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_BYTE                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_WORD                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_HOST_NOTIFY                            == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_32                               == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_64                               == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP                     == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_RESET_DEVICE                       == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS                     == pxSMBusInstance->xProtocol ) )
            {
                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorByte( pxSMBusInstance->pxSMBusProfile,
                                                            pxSMBusInstance->ucCommand, SMBUS_FALSE ) )
                {
                    /* If SMBus_Controller_Write_Descriptor_Byte(  ) is true pop out and wait on descriptor fifo emptying */
                }
                else
                {
                    vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                }
            }

            if( SMBUS_PROTOCOL_SEND_BYTE == pxSMBusInstance->xProtocol )
            {
                vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
            }
            break;
        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerWriteByte
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerWriteByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile    = NULL;
    uint8_t             ucNoStatusCheck = SMBUS_TRUE;
    uint8_t             ucCurrentFill   = 0;
    int                 i               = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_SEND_NEXT_BYTE:
        case E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_WRITE                            == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_BYTE                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_WORD                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_32                               == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_WRITE_64                               == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_SEND_BYTE                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_HOST_NOTIFY                            == pxSMBusInstance->xProtocol ) ||
                ( I2C_PROTOCOL_WRITE                                    == pxSMBusInstance->xProtocol ) ||
                ( I2C_PROTOCOL_WRITE_READ                                == pxSMBusInstance->xProtocol ) )
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usSendDataSize, __LINE__ );

                if( 1 < pxSMBusInstance->usSendDataSize )
                {
                    /* Read how much room is in Descriptor FIFO */
                    ucCurrentFill = ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                    {
                        if( 1 < pxSMBusInstance->usSendDataSize )
                        {
                            ucNoStatusCheck = SMBUS_TRUE;
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->usSendIndex, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex], __LINE__ );
                            if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorByte( pxSMBusInstance->pxSMBusProfile,
                                    pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex], ucNoStatusCheck ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usSendIndex++;
                                pxSMBusInstance->usSendDataSize--;
                            }
                        }
                    }

                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                }
                else
                {
                    vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                    pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                    pxSMBusInstance->usSendDataSize, __LINE__ );
                    if( ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                        ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                        ( I2C_PROTOCOL_WRITE_READ                               == pxSMBusInstance->xProtocol ))
                    {
                        if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorByte( pxSMBusInstance->pxSMBusProfile,
                                pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex], SMBUS_FALSE ) )
                        {
                            /* If SMBus_Controller_Write_Descriptor_Byte(  ) is true
                               pop out and wait on descriptor fifo emptying */
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                        }
                        else
                        {
                            vSMBusHandleActionCreateEventSendNextByte( pxSMBusInstance );
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_SEND_READ_START );
                        }
                    }
                    else
                    {
                        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                        pxSMBusInstance->usSendDataSize, __LINE__ );
                        if( SMBUS_PROTOCOL_HOST_NOTIFY == pxSMBusInstance->xProtocol )    /* No PEC with Host Notify */
                        {
                            if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorStopWrite( pxSMBusInstance->pxSMBusProfile,
                                    pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex] ) )
                            {
                                /* If SMBus_Controller_Write_Descriptor_Byte(  ) is true
                                   pop out and wait on descriptor fifo emptying */
                                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                            }
                            else
                            {
                                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                            }
                        }
                        else
                        {
                            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                        pxSMBusInstance->ucPecRequiredForTransaction, __LINE__ );
                            if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
                            {
                                /* Need space for 2 bytes in decscriptor fifo here */
                                if( SMBUS_FIFO_SPACE_FOR_TWO_BYTES >= ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile ) )
                                {

                                    if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorByte( pxSMBusInstance->pxSMBusProfile,
                                                                            pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex], SMBUS_FALSE ) )
                                    {
                                        /* Should never get here but if so just stay in this state */
                                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                                    }
                                    else
                                    {
                                        ucSMBusControllerWriteDescriptorPECWrite( pxSMBusInstance->pxSMBusProfile );
                                        vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                                    }
                                }
                                else
                                {
                                    /* If SMBus_Controller_Write_Descriptor_Byte(  ) is true
                                       pop out and wait on descriptor fifo emptying */
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                                }
                            }
                            else
                            {
                                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorStopWrite( pxSMBusInstance->pxSMBusProfile,
                                    pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex] ) )
                                {
                                    /* If SMBus_Controller_Write_Descriptor_Byte(  ) is true
                                    pop out and wait on descriptor fifo emptying */
                                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_WRITE_BYTE );
                                }
                                else
                                {
                                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
                                }
                            }
                        }
                    }
                }
            }

            if( SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS == pxSMBusInstance->xProtocol )
            {
                /* Read how much room is in Descriptor FIFO */
                ucCurrentFill = ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                {
                    if( 0 < pxSMBusInstance->usSendDataSize )
                    {
                        ucNoStatusCheck = SMBUS_TRUE;
                        if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerWriteDescriptorByte( pxSMBusInstance->pxSMBusProfile,
                                pxSMBusInstance->ucSendData[pxSMBusInstance->usSendIndex], ucNoStatusCheck ) )
                        {
                            break;
                        }
                        else
                        {
                            pxSMBusInstance->usSendIndex++;
                            pxSMBusInstance->usSendDataSize--;
                        }
                    }
                    else
                    {
                        break;
                    }
                }

                /* Now send the PEC */
                ucSMBusControllerWriteDescriptorPECWrite( pxSMBusInstance->pxSMBusProfile );

                /* And enable the controller to send descriptors */
                vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );

                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
            }

            if( ( SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP         == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_RESET_DEVICE           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED  == pxSMBusInstance->xProtocol ) )
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->xProtocol, __LINE__ );
                ucSMBusControllerWriteDescriptorPECWrite( pxSMBusInstance->pxSMBusProfile );
                vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_AWAITING_DONE );
            }

            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerSendReadStart
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerSendReadStart( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;
    int                 i            = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch ( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_SEND_NEXT_BYTE:
            if( I2C_PROTOCOL_WRITE_READ == pxSMBusInstance->xProtocol )
            {
                pxSMBusInstance->usDescriptorsSent = 0;
                ucSMBusControllerReadDescriptorStart( pxSMBusInstance->pxSMBusProfile,
                                                        pxSMBusInstance->ucSMBusDestinationAddress );
                pxSMBusInstance->ucExpectedByteCountPart =
                ( pxSMBusInstance->usExpectedByteCount < SMBUS_HALF_FIFO_DEPTH ? pxSMBusInstance->usExpectedByteCount : SMBUS_HALF_FIFO_DEPTH );

                /* Set to no more than 32 */
                if( 0 < pxSMBusInstance->ucExpectedByteCountPart ) /* If its zero leave threshold at 1 */
                {
                    vSMBusHWWriteCtrlRxFifoFillThreshold( pxSMBusInstance->pxSMBusProfile,
                                                                pxSMBusInstance->ucExpectedByteCountPart );
                }

                if( 0 == pxSMBusInstance->usExpectedByteCount )
                {
                    ucSMBusControllerReadDescriptorStop( pxSMBusInstance->pxSMBusProfile );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_DONE );
                }
                else
                {
                    //ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_BYTE );
                }
            }



            if( ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) )
            {
                pxSMBusInstance->usDescriptorsSent = 0;
                ucSMBusControllerReadDescriptorStart( pxSMBusInstance->pxSMBusProfile,
                                                        pxSMBusInstance->ucSMBusDestinationAddress );
                ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );    /* Send all descriptors */
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->xProtocol, __LINE__ );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_BLOCK_SIZE );
            }

            if( ( SMBUS_PROTOCOL_READ_64        == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32        == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD      == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE      == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL   == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_RECEIVE_BYTE   == pxSMBusInstance->xProtocol ) )
            {
                switch ( pxSMBusInstance->xProtocol )
                {
                case SMBUS_PROTOCOL_READ_64:
                    pxSMBusInstance->usExpectedByteCount = 8;
                    pxSMBusInstance->ucExpectedByteCountPart = 8;
                    pxSMBusInstance->usDescriptorsSent = 0;
                    break;

                case SMBUS_PROTOCOL_READ_32:
                    pxSMBusInstance->usExpectedByteCount = 4;
                    pxSMBusInstance->ucExpectedByteCountPart = 4;
                    pxSMBusInstance->usDescriptorsSent = 0;
                    break;

                case SMBUS_PROTOCOL_READ_WORD:
                    pxSMBusInstance->usExpectedByteCount = 2;
                    pxSMBusInstance->ucExpectedByteCountPart = 2;
                    pxSMBusInstance->usDescriptorsSent = 0;
                    break;

                case SMBUS_PROTOCOL_READ_BYTE:
                    pxSMBusInstance->usExpectedByteCount = 1;
                    pxSMBusInstance->ucExpectedByteCountPart = 1;
                    pxSMBusInstance->usDescriptorsSent = 0;
                    break;

                case SMBUS_PROTOCOL_PROCESS_CALL:
                    pxSMBusInstance->usExpectedByteCount = 2;
                    pxSMBusInstance->ucExpectedByteCountPart = 2;
                    pxSMBusInstance->usDescriptorsSent = 0;
                    break;

                case SMBUS_PROTOCOL_RECEIVE_BYTE:
                    pxSMBusInstance->usExpectedByteCount = 1;
                    pxSMBusInstance->ucExpectedByteCountPart = 1;
                    pxSMBusInstance->usDescriptorsSent = 0;
                    break;

                default:
                    break;
                }
                vSMBusHWWriteCtrlRxFifoFillThreshold( pxSMBusInstance->pxSMBusProfile,
                                                            pxSMBusInstance->ucExpectedByteCountPart );

                /* WRITE_START, WRITE_BYTE, READ_START, */
                ucSMBusControllerReadDescriptorStart( pxSMBusInstance->pxSMBusProfile,
                                                        pxSMBusInstance->ucSMBusDestinationAddress );

                /* If PEC
                   READ_BYTE * ( BYTE_COUNT ), READ_PEC */

                /* else
                   READ_BYTE * ( BYTE_COUNT-1 ), READ_STOP */

                if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
                {
                    for( i = 0; i < pxSMBusInstance->ucExpectedByteCountPart; i++ )
                    {
                        ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE );
                        pxSMBusInstance->usDescriptorsSent++;
                    }

                    ucSMBusControllerReadDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                }
                else
                {
                    for( i = 0; i < pxSMBusInstance->ucExpectedByteCountPart - 1; i++ )
                    {
                        ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE );
                        pxSMBusInstance->usDescriptorsSent++;
                    }
                    ucSMBusControllerReadDescriptorStop( pxSMBusInstance->pxSMBusProfile );
                    pxSMBusInstance->usDescriptorsSent++;
                }
                vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );   /* Send all descriptors */
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_BYTE );
            }

            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadBlockSize
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadBlockSize( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ:
            /* Need block size before we can proceed */
            break;

        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            ulSMBusHWReadPHYCtrlDbgState( pxTheProfile ), __LINE__ );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            ulSMBusHWReadCtrlDbgState( pxTheProfile ), __LINE__ );

            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            ulSMBusHWReadPHYCtrlDbgState( pxTheProfile ), __LINE__ );
            vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                            pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                            ulSMBusHWReadCtrlDbgState( pxTheProfile ), __LINE__ );
            break;

        case E_CONTROLLER_DATA_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) )
            {
                pxSMBusInstance->usExpectedByteCount = ulSMBusHWReadCtrlRxFifoPayload( pxSMBusInstance->pxSMBusProfile );

                pxSMBusInstance->usDescriptorsSent = 0;
                pxSMBusInstance->ucExpectedByteCountPart =
                    ( pxSMBusInstance->usExpectedByteCount < SMBUS_HALF_FIFO_DEPTH ? pxSMBusInstance->usExpectedByteCount : SMBUS_HALF_FIFO_DEPTH );

                /* Set to no more than 32 */
                if( 0 < pxSMBusInstance->ucExpectedByteCountPart ) /* If its zero leave threshold at 1 */
                {
                    vSMBusHWWriteCtrlRxFifoFillThreshold( pxSMBusInstance->pxSMBusProfile,
                                                                pxSMBusInstance->ucExpectedByteCountPart );
                }

                /* ACK the block read */
                if( ( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction ) ||
                      ( SMBUS_DEVICE_DEFAULT_ARP_ADDRESS == pxSMBusInstance->ucSMBusDestinationAddress ) ) /* PEC required for all ARP protocols */
                {
                    ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_BYTE );

                    if( 0 == pxSMBusInstance->usExpectedByteCount )
                    {
                        /* For the special case of 0 size block with a PEC */
                        ucSMBusControllerReadDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                    }
                }
                else
                {
                    if( 0 == pxSMBusInstance->usExpectedByteCount )
                    {
                        ucSMBusControllerReadDescriptorStop( pxSMBusInstance->pxSMBusProfile );
                        vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_DONE );
                    }
                    else
                    {
                        ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, SMBUS_FALSE );
                        vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_BYTE );
                    }
                }
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadByte
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadByte( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile    = NULL;
    int                 i               = 0;
    uint8_t             ucCurrentFill   = 0;
    uint8_t             ucNoStatusCheck = SMBUS_TRUE;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch ( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) ||
                ( I2C_PROTOCOL_READ                                     == pxSMBusInstance->xProtocol ) ||
                ( I2C_PROTOCOL_WRITE_READ                               == pxSMBusInstance->xProtocol ))
            {
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usDescriptorsSent, __LINE__ );
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usExpectedByteCount, __LINE__ );

                if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
                {
                    if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
                    {
                        ucCurrentFill = ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                        for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                        {
                            if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
                            {
                                ucNoStatusCheck = SMBUS_TRUE;
                                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                                {
                                    break;
                                }
                                else
                                {
                                    pxSMBusInstance->usDescriptorsSent++;
                                }
                            }
                            else
                            {
                                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorPEC( pxSMBusInstance->pxSMBusProfile ) )
                                {
                                }
                                else
                                {
                                    pxSMBusInstance->usDescriptorsSent++;
                                }
                                break;
                            }
                        }
                    }
                    else
                    {
                        ucCurrentFill = ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                        for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                        {
                            if( pxSMBusInstance->usDescriptorsSent < ( pxSMBusInstance->usExpectedByteCount - 1 ) )
                            {
                                ucNoStatusCheck = SMBUS_TRUE;
                                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                                {
                                    break;
                                }
                                else
                                {
                                    pxSMBusInstance->usDescriptorsSent++;
                                }
                            }
                            else if( pxSMBusInstance->usDescriptorsSent == ( pxSMBusInstance->usExpectedByteCount - 1 ) )
                            {
                                if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorStop( pxSMBusInstance->pxSMBusProfile ) )
                                {
                                    break;
                                }
                                else
                                {
                                    pxSMBusInstance->usDescriptorsSent++;
                                }
                            }
                        }
                    }
                    ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                }
            }
            break;

        case E_CONTROLLER_DONE_IRQ:
            /* Got this early just put the event back on the queue until we have read any received data
               SMBus_GenerateEvent_E_CONTROLLER_DONE_IRQ( pxSMBusInstance ); */
            break;

        case E_CONTROLLER_DATA_IRQ:
            if( ( SMBUS_PROTOCOL_READ_64                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_RECEIVE_BYTE                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                  == pxSMBusInstance->xProtocol ) ||
                ( I2C_PROTOCOL_READ                                     == pxSMBusInstance->xProtocol ) ||
                ( I2C_PROTOCOL_WRITE_READ                               == pxSMBusInstance->xProtocol ) )
            {
                /* Read until FIFO is empty. If true RX FIFO is empty */
                while( !ulSMBusHWReadCtrlRxFifoStatusEmpty( pxSMBusInstance->pxSMBusProfile ) )
                {
                    pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex++] =
                        ulSMBusHWReadCtrlRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                }

                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usReceiveIndex, __LINE__ );

                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usExpectedByteCount, __LINE__ );

                if( pxSMBusInstance->usReceiveIndex >= pxSMBusInstance->usExpectedByteCount )
                {
                    /* We've got all the expected data but we might still be expecting a PEC */

                    vSMBusHWWriteCtrlRxFifoFillThreshold( pxSMBusInstance->pxSMBusProfile, 1 );

                    if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
                    {
                        if( pxSMBusInstance->usReceiveIndex > pxSMBusInstance->usExpectedByteCount )
                        {
                            /* We've already got the PEC */
                            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_DONE );
                        }
                    }
                    else
                    {
                        vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_DONE );
                    }
                }
                else
                {
                    /* We are expecting more data - stay in this state and wait on another data IRQ
                        Set threshold to data we are still expecting */
                    if( SMBUS_HALF_FIFO_DEPTH > ( pxSMBusInstance->usExpectedByteCount - pxSMBusInstance->usReceiveIndex ) )
                    {
                        pxSMBusInstance->ucExpectedByteCountPart =
                            ( pxSMBusInstance->usExpectedByteCount - pxSMBusInstance->usReceiveIndex );
                    }
                    else
                    {
                        pxSMBusInstance->ucExpectedByteCountPart = SMBUS_HALF_FIFO_DEPTH;
                    }

                    vSMBusHWWriteCtrlRxFifoFillThreshold( pxSMBusInstance->pxSMBusProfile,
                                                                pxSMBusInstance->ucExpectedByteCountPart );
                }
            }

            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadPEC
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadPEC( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile = NULL;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_DATA_IRQ:
            if( ( SMBUS_PROTOCOL_BLOCK_READ                             == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_64                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_32                                == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_WORD                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_READ_BYTE                              == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_PROCESS_CALL                           == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL    == pxSMBusInstance->xProtocol ) ||
                ( SMBUS_PROTOCOL_RECEIVE_BYTE                           == pxSMBusInstance->xProtocol ) )
            {
                if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
                {
                    ucSMBusControllerReadDescriptorPEC( pxSMBusInstance->pxSMBusProfile );
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                    vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_CONTROLLER_READ_DONE );
                }
            }
            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Handle any event to the State Machine
*           when the current state is ControllerReadDone
*
*****************************************************************************/
static void vDefaultSMBusFSMSMBusStateControllerReadDone( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    SMBUS_PROFILE_TYPE* pxTheProfile    = NULL;
    uint8_t             ucNoStatusCheck = SMBUS_TRUE;
    uint8_t             ucCurrentFill   = 0;
    int                 i               = 0;

    if( NULL != pxSMBusInstance )
    {
        pxTheProfile = pxSMBusInstance->pxSMBusProfile;

        switch( ucAnyEvent )
        {
        case E_CONTROLLER_DATA_IRQ:
            if( I2C_PROTOCOL_READ == pxSMBusInstance->xProtocol )
            {
                /* Read until FIFO is empty. If true RX FIFO is empty */
                while( !ulSMBusHWReadCtrlRxFifoStatusEmpty( pxSMBusInstance->pxSMBusProfile ) )
                {
                    pxSMBusInstance->ucReceivedData[pxSMBusInstance->usReceiveIndex++] =
                        ulSMBusHWReadCtrlRxFifoPayload( pxSMBusInstance->pxSMBusProfile );
                }

                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usReceiveIndex, __LINE__ );

                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->usExpectedByteCount, __LINE__ );

            }
        break;

        case E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ:
            /* Check if we need to send any more ACKs */
            if( I2C_PROTOCOL_READ == pxSMBusInstance->xProtocol )
            {
                if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usReceiveIndex )
                {
                    if( pxSMBusInstance->usDescriptorsSent == ( pxSMBusInstance->usReceiveIndex-1 ) )
                    {
                        if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorStop( pxSMBusInstance->pxSMBusProfile ) )
                        {
                            break;
                        }
                        else
                        {
                            pxSMBusInstance->usDescriptorsSent++;
                            vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                        }
                    }
                    else
                    {
                        if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                        {
                            break;
                        }
                        else
                        {
                            pxSMBusInstance->usDescriptorsSent++;
                            vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                        }
                    }

                }
            }
            else if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
            {
                if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
                {
                    ucCurrentFill = ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                    {
                        if( pxSMBusInstance->usDescriptorsSent < pxSMBusInstance->usExpectedByteCount )
                        {
                            ucNoStatusCheck = SMBUS_TRUE;
                            if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                        }
                        else
                        {
                            if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorPEC( pxSMBusInstance->pxSMBusProfile ) )
                            {
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                            break;
                        }
                    }
                }
                else
                {
                    ucCurrentFill = ulSMBusHWReadCtrlDescStatusFillLevel( pxSMBusInstance->pxSMBusProfile );
                    for( i = 0; i < ( SMBUS_FIFO_DEPTH - ucCurrentFill ); i++ )
                    {
                        if( pxSMBusInstance->usDescriptorsSent < ( pxSMBusInstance->usExpectedByteCount - 1 ) )
                        {
                            ucNoStatusCheck = SMBUS_TRUE;
                            if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorByte( pxSMBusInstance->pxSMBusProfile, ucNoStatusCheck ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                        }
                        else if( pxSMBusInstance->usDescriptorsSent == ( pxSMBusInstance->usExpectedByteCount - 1 ) )
                        {
                            if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorStop( pxSMBusInstance->pxSMBusProfile ) )
                            {
                                break;
                            }
                            else
                            {
                                pxSMBusInstance->usDescriptorsSent++;
                            }
                        }
                    }
                    vSMBusHWWriteCtrlControlEnable( pxSMBusInstance->pxSMBusProfile, SMBUS_CONTROL_ENABLE );
                }
            }
            else if( SMBUS_TRUE == pxSMBusInstance->ucPecRequiredForTransaction )
            {
                if( pxSMBusInstance->usDescriptorsSent == pxSMBusInstance->usExpectedByteCount )
                {
                    /* Descriptor for PEC still required */
                    if( SMBUS_HW_DESCRIPTOR_WRITE_FAIL == ucSMBusControllerReadDescriptorPEC( pxSMBusInstance->pxSMBusProfile ) )
                    {
                    }
                    else
                    {
                        pxSMBusInstance->usDescriptorsSent++;
                    }
                }
            }
            break;

        case E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ:
        case E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusWarning( pxSMBusInstance, ucAnyEvent );
            break;

        case E_CONTROLLER_LOA_ERROR_IRQ:
        case E_CONTROLLER_NACK_ERROR_IRQ:
        case E_CONTROLLER_PEC_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_CONTROLLER_DESC_ERROR_IRQ:
            /* Report Result to Application */
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID, SMBUS_ERROR );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_TARGET_LOA_ERROR_IRQ:
        case E_TARGET_PEC_ERROR_IRQ:
        case E_TARGET_RX_FIFO_ERROR_ERROR_IRQ:
        case E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ:
        case E_TARGET_DESC_ERROR_IRQ:
        case E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ:
        case E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ:
        case E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ:
            vSMBusHandleActionBusError( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;

        case E_CONTROLLER_DONE_IRQ:
            switch( pxSMBusInstance->xProtocol )
            {
            case SMBUS_PROTOCOL_BLOCK_READ:
            case SMBUS_PROTOCOL_READ_64:
            case SMBUS_PROTOCOL_READ_32:
            case SMBUS_PROTOCOL_READ_WORD:
            case SMBUS_PROTOCOL_READ_BYTE:
            case SMBUS_PROTOCOL_PROCESS_CALL:
            case SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL:
            case SMBUS_PROTOCOL_RECEIVE_BYTE:
            case SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED:
            case SMBUS_ARP_PROTOCOL_GET_UDID:
            {
                /* Report Data to Application */
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->xProtocol, __LINE__ );
                vSMBusHandleActionWriteDataToApplication( pxSMBusInstance, pxTheProfile->ulTransactionID );
                vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance,
                                                                    pxTheProfile->ulTransactionID, SMBUS_SUCCESS );

                /* Log completed message */
                pxSMBusInstance->ulMessagesComplete[pxSMBusInstance->xProtocol]++;

                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            break;

            case I2C_PROTOCOL_READ:
            case I2C_PROTOCOL_WRITE_READ:
            {
                /* Report Data to Application */
                vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                                pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                                pxSMBusInstance->xProtocol, __LINE__ );
                vSMBusHandleActionWriteI2CDataToApplication( pxSMBusInstance );
                vSMBusHandleActionAnnounceI2CResultToApplication( pxSMBusInstance, SMBUS_SUCCESS );

                /* Log completed message */
                pxSMBusInstance->ulMessagesComplete[pxSMBusInstance->xProtocol]++;

                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            break;

            case SMBUS_PROTOCOL_SEND_BYTE:
            case SMBUS_PROTOCOL_WRITE_BYTE:
            case SMBUS_PROTOCOL_WRITE_WORD:
            case SMBUS_PROTOCOL_BLOCK_WRITE:
            case SMBUS_PROTOCOL_HOST_NOTIFY:
            case SMBUS_PROTOCOL_WRITE_32:
            case SMBUS_PROTOCOL_WRITE_64:
            case SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP:
            case SMBUS_ARP_PROTOCOL_RESET_DEVICE:
            case SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS:
            case SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED:
            {
                /* Report Result to Application */
                vSMBusHandleActionAnnounceResultToApplication( pxSMBusInstance,
                                                                    pxTheProfile->ulTransactionID, SMBUS_SUCCESS );

                /* Log completed message */
                pxSMBusInstance->ulMessagesComplete[pxSMBusInstance->xProtocol]++;

                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            break;

            case I2C_PROTOCOL_WRITE:
            {
                /* Report Result to Application */
                vSMBusHandleActionAnnounceI2CResultToApplication( pxSMBusInstance, SMBUS_SUCCESS );

                /* Log completed message */
                pxSMBusInstance->ulMessagesComplete[pxSMBusInstance->xProtocol]++;

                vSMBusHandleActionResetAllData( pxSMBusInstance );
                vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            }
            break;

            default:
                break;
            }

            break;

        default:
            vSMBusLogUnexpected( pxSMBusInstance, ucAnyEvent );
            vSMBusHandleActionResetAllData( pxSMBusInstance );
            vSMBusNextStateDecoder( pxSMBusInstance, SMBUS_STATE_INITIAL );
            break;
        }
    }
}

/******************************************************************************
*
* @brief    Is the finite state machine for the specified SMBus instance
*           The function will look up the current state, bytes sent, bytes received etc
*           and given the event passed in it will transition to a new state
*
*****************************************************************************/
void vSMBusFSM( SMBUS_INSTANCE_TYPE* pxSMBusInstance, uint8_t ucAnyEvent )
{
    if( NULL != pxSMBusInstance )
    {
        vSMBusClearAction( pxSMBusInstance );

        SMBus_State_Type xState = pxSMBusInstance->xState;
        pxSMBusInstance->ucEvent = ucAnyEvent;

        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_INFO,
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_FSM_EVENT,
                        ( uint32_t )xState, ( uint32_t )ucAnyEvent );
        vLogAddEntry( pxSMBusInstance->pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                        pxSMBusInstance->ucThisInstanceNumber, SMBUS_LOG_EVENT_DEBUG,
                        ( uint32_t )ucAnyEvent, __LINE__ );

        switch ( pxSMBusInstance->xState )
        {
        case SMBUS_STATE_INITIAL:
            vDefaultSMBusFSMSMBusStateInitial( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_AWAITING_COMMAND_BYTE:
            vDefaultSMBusFSMSMBusStateAwaitingCommandByte( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_AWAITING_BLOCK_SIZE:
            vDefaultSMBusFSMSMBusStateAwaitingBlockSize( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_AWAITING_DATA:
            vDefaultSMBusFSMSMBusStateAwaitingData( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_AWAITING_READ:
            vDefaultSMBusFSMSMBusStateAwaitingRead( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_READY_TO_SEND_BYTE:
            vDefaultSMBusFSMSMBusStateReadyToSendByte( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CHECK_IF_PEC_REQUIRED:
            vDefaultSMBusFSMSMBusStateCheckIfPECRequired( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_AWAITING_DONE:
            vDefaultSMBusFSMSMBusStateAwaitingDone( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_SEND_COMMAND:
            vDefaultSMBusFSMSMBusStateControllerSendCommand( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_SEND_READ_START:
            vDefaultSMBusFSMSMBusStateControllerSendReadStart( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_READ_BLOCK_SIZE:
            vDefaultSMBusFSMSMBusStateControllerReadBlockSize( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_READ_BYTE:
            vDefaultSMBusFSMSMBusStateControllerReadByte( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_READ_PEC:
            vDefaultSMBusFSMSMBusStateControllerReadPEC( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_READ_DONE:
            vDefaultSMBusFSMSMBusStateControllerReadDone( pxSMBusInstance, ucAnyEvent );
            break;

        case SMBUS_STATE_CONTROLLER_WRITE_BYTE:
            vDefaultSMBusFSMSMBusStateControllerWriteByte( pxSMBusInstance, ucAnyEvent );
            break;

        default:
            break;
        }
    }
}
